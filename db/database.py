from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["control_narrative_ai"]
