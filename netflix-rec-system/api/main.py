import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models.hybrid import HybridRecommender
from cache.redis_client import get_cached_recs, set_cached_recs

model = HybridRecommender()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    model.train()
    print("Models ready.")
    yield

app = FastAPI(title="Netflix-style Rec Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/recommend/{user_id}")
async def recommend(
    user_id: str,
    last_item:     str  = Query(default="m1"),
    top_n:         int  = Query(default=10, le=50),
    force_refresh: bool = Query(default=False)
):
    if not force_refresh:
        cached = get_cached_recs(user_id)
        if cached:
            return {"user_id": user_id, "source": "cache", "recommendations": cached}

    start = time.perf_counter()
    recs  = model.recommend(user_id, last_item, top_n)
    ms    = round((time.perf_counter() - start) * 1000, 2)

    set_cached_recs(user_id, recs, ttl=300)
    return {
        "user_id": user_id,
        "source": "model",
        "latency_ms": ms,
        "recommendations": recs
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
