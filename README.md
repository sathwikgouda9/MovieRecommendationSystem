A production-grade recommendation engine using Neural Collaborative Filtering, MongoDB Change Streams, and Redis caching built with Python, TensorFlow, and FastAPI.

Features

 Neural CF — GMF + MLP deep learning model for personalized recommendations
 Content-Based Filtering — TF-IDF + cosine similarity on movie genres
 Hybrid Fusion — Weighted ensemble of both models
 Real-Time Updates — MongoDB Change Streams invalidate cache on new ratings
 Redis Caching — Pre-computed recs served in milliseconds
 Dockerized — One command spins up the full stack


Quick Start
bash# Start all services
docker-compose up --build

# Seed data and train models
docker-compose exec app python train.py

# Get recommendations
curl http://localhost:8000/recommend/u1?last_item=m5&top_n=10

API
MethodEndpointDescriptionGET/recommend/{user_id}Get recommendationsGET/healthHealth check
Query params: last_item · top_n (max 50) · force_refresh
Example response:
json{
  "user_id": "u1",
  "source": "model",
  "latency_ms": 12.4,
  "recommendations": ["m87", "m203", "m14", "m55", "m301"]
}

Tech Stack
TensorFlow · MongoDB · Redis · FastAPI · scikit-learn · Docker
