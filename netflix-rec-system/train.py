import random
import datetime
from db.mongo_client import users_col, items_col, interactions_col
from models.hybrid import HybridRecommender

def seed():
    random.seed(42)
    users = [
        {"user_id": f"u{i}", "name": f"User{i}",
         "preferences": random.sample(["action","drama","sci-fi","comedy","thriller"], 2)}
        for i in range(1, 201)
    ]
    items = [
        {"item_id": f"m{i}", "title": f"Movie{i}",
         "genres": random.sample(["action","drama","sci-fi","comedy","thriller"], 2),
         "avg_rating": round(random.uniform(2.5, 5.0), 1)}
        for i in range(1, 501)
    ]
    interactions = [
        {"user_id": f"u{random.randint(1, 200)}",
         "item_id": f"m{random.randint(1, 500)}",
         "rating":  random.randint(1, 5),
         "timestamp": datetime.datetime.utcnow()}
        for _ in range(10000)
    ]
    users_col.delete_many({})
    items_col.delete_many({})
    interactions_col.delete_many({})
    users_col.insert_many(users)
    items_col.insert_many(items)
    interactions_col.insert_many(interactions)
    print("✅ Seeded: 200 users, 500 movies, 10,000 interactions")

if __name__ == "__main__":
    print("Seeding MongoDB...")
    seed()
    print("Training models...")
    rec = HybridRecommender()
    rec.train()
    print("Done. Models saved to saved_models/")
