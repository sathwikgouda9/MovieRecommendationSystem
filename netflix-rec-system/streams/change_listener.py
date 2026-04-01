import os
import datetime
from pymongo import MongoClient
from cache.redis_client import get_redis

def invalidate_user_cache(user_id: str):
    r   = get_redis()
    key = f"recs:{user_id}"
    if r.exists(key):
        r.delete(key)
        print(f"[Stream] Cache busted for {user_id}")

def update_user_profile(user_id: str, item_id: str, rating: int, db):
    db.user_profiles.update_one(
        {"user_id": user_id},
        {
            "$push": {
                "recent_interactions": {
                    "$each":  [{"item_id": item_id, "rating": rating}],
                    "$slice": -50   # keep last 50 interactions
                }
            },
            "$set": {"last_active": datetime.datetime.utcnow()}
        },
        upsert=True
    )

def listen():
    client   = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db       = client["recommendation_db"]
    pipeline = [{"$match": {"operationType": {"$in": ["insert", "update"]}}}]

    print("[Stream] Listening for interaction changes...")
    with db.interactions.watch(pipeline, full_document="updateLookup") as stream:
        for change in stream:
            doc     = change.get("fullDocument", {})
            user_id = doc.get("user_id")
            item_id = doc.get("item_id")
            rating  = doc.get("rating", 0)

            if user_id and item_id:
                invalidate_user_cache(user_id)
                update_user_profile(user_id, item_id, rating, db)
                print(f"[Stream] user={user_id} item={item_id} rating={rating}")

if __name__ == "__main__":
    listen()
