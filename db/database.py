from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["control_narrative_ai"]

users_collection = db["users"]
documents_collection = db["documents"]
parsed_documents_collection = db["parsed_documents"]

chat_sessions_collection = db["chat_sessions"]
chat_messages_collection = db["chat_messages"]