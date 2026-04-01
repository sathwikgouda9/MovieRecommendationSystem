import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db.mongo_client import items_col

class ContentBasedFilter:
    def __init__(self):
        self.tfidf      = TfidfVectorizer(stop_words="english")
        self.sim_matrix = None
        self.items_df   = None

    def train(self):
        items = list(items_col.find({}, {"_id": 0, "item_id": 1, "title": 1, "genres": 1}))
        self.items_df         = pd.DataFrame(items)
        self.items_df["soup"] = self.items_df["genres"].apply(lambda x: " ".join(x))
        tfidf_matrix          = self.tfidf.fit_transform(self.items_df["soup"])
        self.sim_matrix       = cosine_similarity(tfidf_matrix)

    def recommend(self, item_id, top_n=20):
        if self.items_df is None:
            return []
        idx_map = {v: i for i, v in enumerate(self.items_df["item_id"])}
        if item_id not in idx_map:
            return []
        idx    = idx_map[item_id]
        scores = sorted(enumerate(self.sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
        return [(self.items_df.iloc[i]["item_id"], float(s)) for i, s in scores]
