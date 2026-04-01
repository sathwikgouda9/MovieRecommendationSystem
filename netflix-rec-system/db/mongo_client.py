from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["recommendation_db"]

users_col        = db["users"]
items_col        = db["items"]
interactions_col = db["interactions"]
