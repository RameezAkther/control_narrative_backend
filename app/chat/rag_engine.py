import os
import json
import asyncio
from bson import ObjectId
from typing import List, Tuple, Optional, Any

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool

# --- CREWAI IMPORTS ---
from crewai import Agent, Task, Crew, Process

# Database Imports
from db.database import parsed_documents_collection, chat_messages_collection

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Mapping UI selections to Database Fields
ARTIFACT_MAPPING = {
    "Loop Map": "mapper_agent_output_file_path",
    "Logic Extracted": "control_logic_agent_output_file_path",
    "Validation": "validator_agent_output_file_path",
    "PLC Code": "code_generator_agent_output_file_path"
}

MAP_REDUCE_CHAR_THRESHOLD = 40000 

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 1. Main Reasoning LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True 
        )

        # 2. Summarizer & Fast LLM
        self.summarizer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )

    # ... [Keep _get_document_paths, _get_artifact_context, _retrieve_context, _format_history unchanged] ...
    
    def _get_document_paths(self, document_ids: List[str]) -> List[dict]:
        paths = []
        valid_obj_ids = [ObjectId(d) for d in document_ids if ObjectId.is_valid(d)]
        if not valid_obj_ids: return []
        results = parsed_documents_collection.find({"document_id": {"$in": valid_obj_ids}}, {"embeddings_info": 1})
        for doc in results:
            if "embeddings_info" in doc:
                paths.append({"path": doc["embeddings_info"]["persist_directory"], "collection": doc["embeddings_info"]["collection"]})
        return paths

    def _get_precomputed_summary(self, document_ids: List[str]) -> str:
        valid_obj_ids = [ObjectId(d) for d in document_ids if ObjectId.is_valid(d)]
        if not valid_obj_ids: return ""
        results = parsed_documents_collection.find({"document_id": {"$in": valid_obj_ids}}, {"understanding_agent_output_file_path": 1})
        combined_summaries = []
        for doc in results:
            file_path = doc.get("understanding_agent_output_file_path")
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        content = data.get("summary", json.dumps(data, indent=2))
                        combined_summaries.append(content)
                except Exception: pass
        return "\n\n".join(combined_summaries)

    def _get_artifact_context(self, document_ids: List[str], artifacts: List[str]) -> Tuple[str, List[dict]]:
        valid_obj_ids = [ObjectId(d) for d in document_ids if ObjectId.is_valid(d)]
        if not valid_obj_ids: return "", []
        projection = {"document_id": 1}
        fields_to_fetch = []
        for art in artifacts:
            if art in ARTIFACT_MAPPING:
                field = ARTIFACT_MAPPING[art]
                projection[field] = 1
                fields_to_fetch.append((art, field))
        results = parsed_documents_collection.find({"document_id": {"$in": valid_obj_ids}}, projection)
        context_parts = []
        retrieved_artifacts = []
        for doc in results:
            for art_name, field_key in fields_to_fetch:
                file_path = doc.get(field_key)
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = json.dumps(json.load(f), indent=2) if file_path.endswith('.json') else f.read()
                        if content:
                            context_parts.append(f"--- START {art_name} ---\n{content}\n--- END {art_name} ---\n")
                            retrieved_artifacts.append({"source": f"{art_name}", "content": content[:300] + "...", "page": 0})
                    except: pass
        return "\n".join(context_parts), retrieved_artifacts

    def _retrieve_context(self, paths: List[dict], query: str, k: int = 4) -> List[Document]:
        aggregated_docs = []
        for p in paths:
            try:
                vectorstore = Chroma(persist_directory=p["path"], collection_name=p["collection"], embedding_function=self.embeddings)
                aggregated_docs.extend(vectorstore.similarity_search(query, k=k))
            except: continue
        return aggregated_docs

    def _format_history(self, context_ids: List[str]) -> List[object]:
        if not context_ids: return []
        valid_obj_ids = [ObjectId(m) for m in context_ids if ObjectId.is_valid(m)]
        if not valid_obj_ids: return []
        msgs = chat_messages_collection.find({"_id": {"$in": valid_obj_ids}}).sort("created_at", 1)
        history = []
        for m in msgs:
            content = m.get("summary") or m["content"]
            history.append(HumanMessage(content=content) if m["role"] == "user" else AIMessage(content=content))
        return history

    async def _generate_short_summary(self, text: str) -> str:
        try:
            chain = ChatPromptTemplate.from_messages([("system", "Summarize in 1-2 sentences."), ("human", "{text}")]) | self.summarizer_llm | StrOutputParser()
            return await chain.ainvoke({"text": text})
        except: return text[:200]

    async def _classify_query_intent(self, query: str) -> str:
        try:
            chain = ChatPromptTemplate.from_messages([("system", "Return 'summary' if the user asks for a document overview, else 'search'."), ("human", "{query}")]) | self.summarizer_llm | StrOutputParser()
            return (await chain.ainvoke({"query": query})).strip().lower()
        except: return "search"

    # --- STRATEGY 1: RE-ACT AGENT (VIA CREWAI) ---
    async def _generate_react_response(self, query: str, doc_paths: List[dict], history: List[object]) -> Tuple[str, List[dict]]:
        print("--- Executing ReACT Mode via CrewAI ---")
        retrieved_docs_log = []

        def search_tool_func(q: str):
            docs = self._retrieve_context(doc_paths, q, k=3)
            for d in docs:
                retrieved_docs_log.append({
                    "content": d.page_content,
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", 0)
                })
            return "\n\n".join([d.page_content for d in docs])

        search_tool = Tool(
            name="Control_Narrative_Search",
            func=search_tool_func,
            description="Useful for searching technical details. Input: search query."
        )

        analyst = Agent(
            role='Senior Technical Analyst',
            goal='Provide accurate technical answers.',
            backstory="You are an expert analyst for industrial control systems.",
            tools=[search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        history_context = "\n".join([f"{type(m).__name__}: {m.content}" for m in history[-3:]])
        task = Task(
            description=f"Analyze the question with history.\nHistory: {history_context}\nQuestion: {query}",
            expected_output="Detailed technical response.",
            agent=analyst
        )

        crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=True)
        try:
            result = await asyncio.to_thread(crew.kickoff)
            return str(result), retrieved_docs_log
        except Exception as e:
            print(f"CrewAI Error: {e}")
            return "I encountered an error with the Agent.", []

    # --- STRATEGY 2: FLARE (Active Retrieval) ---
    async def _generate_flare_response(self, query: str, doc_paths: List[dict], history: List[object]) -> Tuple[str, List[dict]]:
        print("--- Executing Flare Mode ---")
        # 1. Draft
        draft_chain = ChatPromptTemplate.from_messages([("system", "Expert Assistant."), MessagesPlaceholder(variable_name="history"), ("human", "{query}")]) | self.summarizer_llm | StrOutputParser()
        draft_answer = await draft_chain.ainvoke({"history": history, "query": query})

        # 2. Critique
        critique_chain = ChatPromptTemplate.from_messages([("human", "Identify facts needing verification. Return queries one per line, or NONE.\nAnswer: {answer}")]) | self.summarizer_llm | StrOutputParser()
        critique_output = await critique_chain.ainvoke({"answer": draft_answer})
        
        search_queries = [line.strip() for line in critique_output.split('\n') if line.strip() and "NONE" not in line]
        if not search_queries: search_queries = [query]

        # 3. Retrieve
        context_text = ""
        retrieved_docs = []
        for q in search_queries[:3]:
            docs = self._retrieve_context(doc_paths, q, k=2)
            for d in docs:
                context_text += f"\nSnippet for '{q}':\n{d.page_content}\n"
                retrieved_docs.append({"content": d.page_content, "source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", 0)})

        # 4. Refine (FIXED: Using variables to prevent JSON parsing errors)
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine the answer using the verified context."),
            ("human", "Question: {query}\nDraft: {draft}\nContext: {context}\nFinal Answer:")
        ])
        refine_chain = refine_prompt | self.llm | StrOutputParser()
        
        final_answer = await refine_chain.ainvoke({
            "query": query,
            "draft": draft_answer,
            "context": context_text
        })
        return final_answer, retrieved_docs

    # --- STRATEGY 3: MAP REDUCE (Token Overflow) ---
    async def _execute_map_reduce(self, query: str, large_context: str, history: List[object]) -> str:
        print("Context too large. Triggering Map-Reduce...")
        chunks = self.text_splitter.split_text(large_context)
        
        # Map
        map_chain = ChatPromptTemplate.from_messages([("system", "Extract relevant info."), ("human", "Chunk: {chunk}\nQuery: {query}")]) | self.llm | StrOutputParser()
        batch_inputs = [{"chunk": chunk, "query": query} for chunk in chunks]
        partial_results = await map_chain.abatch(batch_inputs)
        
        relevant = [r for r in partial_results if "No relevant info" not in r and len(r) > 5]
        if not relevant: return "No relevant information found in the documents."
        
        combined_partials = "\n---\n".join(relevant)

        # Reduce (FIXED: Using variables to prevent JSON parsing errors)
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize findings."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Partial Findings from Document Chunks:\n{findings}\n\nQuestion: {query}")
        ])
        reduce_chain = reduce_prompt | self.llm
        
        # Pass JSON-heavy strings as inputs, NOT in the template
        response = await reduce_chain.ainvoke({
            "history": history,
            "findings": combined_partials,
            "query": query
        })
        return response.content

    # --- MAIN ENTRY POINT ---
    async def generate_response(
        self, 
        query: str, 
        document_ids: List[str], 
        active_context_ids: List[str],
        artifacts: List[str] = [],
        mode: str = "RAG" 
    ) -> tuple[str, List[dict], str]: 
        
        context_text = ""
        retrieved_docs = []
        response_content = ""
        
        doc_paths = self._get_document_paths(document_ids)
        history_messages = self._format_history(active_context_ids)

        # 1. ARTIFACTS
        if artifacts and len(document_ids) > 0:
            print(f"Using Artifacts: {artifacts}")
            context_text, retrieved_docs = self._get_artifact_context(document_ids, artifacts)
            if not context_text: context_text = "No content found for artifacts."
            
            if len(context_text) > MAP_REDUCE_CHAR_THRESHOLD:
                response_content = await self._execute_map_reduce(query, context_text, history_messages)
            else:
                # FIXED: Use input variables for Context
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer using Artifacts."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "Context:\n{context}\n\nQuestion: {query}")
                ])
                response_content = (await (prompt | self.llm).ainvoke({
                    "history": history_messages,
                    "context": context_text, # JSON string passed safely here
                    "query": query
                })).content

        # 2. ReACT (CrewAI)
        elif mode == "ReACT" and len(doc_paths) > 0:
            response_content, retrieved_docs = await self._generate_react_response(query, doc_paths, history_messages)

        # 3. Flare
        elif mode == "Flare" and len(doc_paths) > 0:
            response_content, retrieved_docs = await self._generate_flare_response(query, doc_paths, history_messages)

        # 4. Standard RAG (Auto/Default)
        else:
            print("--- Executing Standard RAG ---")
            intent = await self._classify_query_intent(query)
            
            if intent == "summary" and len(document_ids) > 0:
                summary = self._get_precomputed_summary(document_ids)
                if summary:
                    context_text = f"SUMMARY:\n{summary}"
                    retrieved_docs = [{"source": "Summary", "content": "Full Summary", "page": 0}]
            
            if not context_text and doc_paths:
                raw_docs = self._retrieve_context(doc_paths, query)
                context_text = "\n".join([d.page_content for d in raw_docs])
                for d in raw_docs: retrieved_docs.append({"content": d.page_content, "source": d.metadata.get("source"), "page": d.metadata.get("page")})

            if not context_text: context_text = "No documents selected."

            if len(context_text) > MAP_REDUCE_CHAR_THRESHOLD:
                response_content = await self._execute_map_reduce(query, context_text, history_messages)
            else:
                # FIXED: Use input variables for Context
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Helpful Assistant."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "Context:\n{context}\n\nQuestion: {query}")
                ])
                response_content = (await (prompt | self.llm).ainvoke({
                    "history": history_messages,
                    "context": context_text, # Safe injection
                    "query": query
                })).content

        response_summary = await self._generate_short_summary(response_content)
        return response_content, retrieved_docs, response_summary

rag_engine = RAGEngine()