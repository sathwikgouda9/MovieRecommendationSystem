from models.ncf_model import NCFModel
from models.content_based import ContentBasedFilter

class HybridRecommender:
    def __init__(self, alpha=0.65):
        self.ncf   = NCFModel()
        self.cb    = ContentBasedFilter()
        self.alpha = alpha  # weight for NCF vs content-based

    def train(self):
        print("Training Neural CF model...")
        self.ncf.train()
        print("Training content-based model...")
        self.cb.train()
        print("All models ready.")

    def recommend(self, user_id, last_item_id, top_n=10):
        ncf_recs = self.ncf.recommend(user_id, top_n=top_n * 2)
        cb_recs  = self.cb.recommend(last_item_id, top_n=top_n * 2)

        scores = {}
        for rank, (item, _) in enumerate(ncf_recs):
            scores[item] = scores.get(item, 0) + self.alpha * (1 / (rank + 1))
        for rank, (item, _) in enumerate(cb_recs):
            scores[item] = scores.get(item, 0) + (1 - self.alpha) * (1 / (rank + 1))

        return sorted(scores, key=scores.get, reverse=True)[:top_n]
