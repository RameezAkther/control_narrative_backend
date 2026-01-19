import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CHANGED IMPORTS START ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
# --- CHANGED IMPORTS END ---

from db.database import parsed_documents_collection, chat_messages_collection
from bson import ObjectId

# --- CONFIGURATION ---
# Ensure this matches the model you used to Create the embeddings
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class RAGEngine:
    def __init__(self):
        # Initialize Local Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", # or gemini-pro
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3, # Low temp for factual RAG
            convert_system_message_to_human=True 
        )

    def _get_document_paths(self, document_ids: List[str]) -> List[dict]:
        paths = []
        valid_obj_ids = []

        # 1. Filter for valid ObjectIds only
        for d_id in document_ids:
            if ObjectId.is_valid(d_id):
                valid_obj_ids.append(ObjectId(d_id))
        
        if not valid_obj_ids:
            return []

        # 2. Query DB with valid IDs
        results = parsed_documents_collection.find(
            {"document_id": {"$in": valid_obj_ids}},
            {"embeddings_info": 1, "document_id": 1}
        )

        for doc in results:
            if "embeddings_info" in doc:
                paths.append({
                    "path": doc["embeddings_info"]["persist_directory"],
                    "collection": doc["embeddings_info"]["collection"]
                })
        return paths

    def _retrieve_context(self, paths: List[dict], query: str, k: int = 4) -> List[Document]:
        """
        Iterates through multiple ChromaDB directories (one per doc) 
        and aggregates relevant chunks.
        """
        aggregated_docs = []
        
        for p in paths:
            try:
                # Load the specific vector store for this document
                vectorstore = Chroma(
                    persist_directory=p["path"],
                    collection_name=p["collection"],
                    embedding_function=self.embeddings
                )
                
                # Search
                results = vectorstore.similarity_search(query, k=k)
                aggregated_docs.extend(results)
            except Exception as e:
                print(f"Error loading vectorstore at {p['path']}: {e}")
                continue
        
        return aggregated_docs

    def _format_history(self, context_ids: List[str]) -> List[object]:
        if not context_ids:
            return []

        # 1. Filter for valid ObjectIds only
        valid_obj_ids = []
        for m_id in context_ids:
            if ObjectId.is_valid(m_id):
                valid_obj_ids.append(ObjectId(m_id))
        
        if not valid_obj_ids:
            return []

        # 2. Fetch messages using only valid IDs
        msgs = chat_messages_collection.find({"_id": {"$in": valid_obj_ids}}).sort("created_at", 1)
        
        history = []
        for m in msgs:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                history.append(AIMessage(content=m["content"]))
        
        return history

    async def generate_response(
        self, 
        query: str, 
        document_ids: List[str], 
        active_context_ids: List[str]
    ) -> tuple[str, List[dict]]: # Return type changed to Tuple
        
        # 1. Retrieve Context
        doc_paths = self._get_document_paths(document_ids)
        retrieved_docs = []
        context_text = ""
        
        if doc_paths:
            raw_docs = self._retrieve_context(doc_paths, query)
            
            # Format docs for Context String AND for Database Storage
            context_text = "\n\n".join([d.page_content for d in raw_docs])
            
            # Create a clean list of dicts to store in MongoDB
            for d in raw_docs:
                retrieved_docs.append({
                    "content": d.page_content,
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", 0),
                    # Add any other metadata you saved during ingestion
                })
        else:
            context_text = "No documents selected."

        # 2. Retrieve History
        history_messages = self._format_history(active_context_ids)

        # 3. Construct Prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an intelligent assistant analyzing technical control narratives. "
                "Use the provided context to answer the user's question accurately. "
                "If the answer is not in the context, state that you cannot find it in the documents."
            )),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}")
        ])

        # 4. Invoke Chain
        chain = prompt | self.llm
        
        response = await chain.ainvoke({
            "history": history_messages
        })

        # Return BOTH the content and the citations list
        return response.content, retrieved_docs

# Singleton instance
rag_engine = RAGEngine()
