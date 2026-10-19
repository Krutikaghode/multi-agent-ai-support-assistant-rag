from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from app.config import MONGODB_URI, MONGODB_DB_NAME

def get_mongodb_checkpointer() -> MongoDBSaver:
    client = MongoClient(MONGODB_URI)
    return MongoDBSaver(client, db_name=MONGODB_DB_NAME)
