import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from pymongo import MongoClient
import os

class NCFModel:
    def __init__(self, embedding_dim=64, mlp_layers=[128, 64, 32]):
        self.embedding_dim = embedding_dim
        self.mlp_layers    = mlp_layers
        self.model         = None
        self.user_encoder  = {}
        self.item_encoder  = {}

    def _build_model(self, n_users, n_items):
        user_input = layers.Input(shape=(1,), name="user_id")
        item_input = layers.Input(shape=(1,), name="item_id")

        # GMF branch
        user_gmf = layers.Flatten()(layers.Embedding(n_users, self.embedding_dim)(user_input))
        item_gmf = layers.Flatten()(layers.Embedding(n_items, self.embedding_dim)(item_input))
        gmf_out  = layers.Multiply()([user_gmf, item_gmf])

        # MLP branch
        user_mlp = layers.Flatten()(layers.Embedding(n_users, self.embedding_dim)(user_input))
        item_mlp = layers.Flatten()(layers.Embedding(n_items, self.embedding_dim)(item_input))
        x = layers.Concatenate()([user_mlp, item_mlp])
        for units in self.mlp_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(0.2)(x)

        # Fusion
        fusion = layers.Concatenate()([gmf_out, x])
        output = layers.Dense(1, activation="sigmoid")(fusion)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def load_data(self):
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        df = pd.DataFrame(list(client["recommendation_db"].interactions.find(
            {}, {"_id": 0, "user_id": 1, "item_id": 1, "rating": 1}
        )))
        df["label"] = (df["rating"] >= 4).astype(int)
        self.user_encoder = {u: i for i, u in enumerate(df["user_id"].unique())}
        self.item_encoder = {m: i for i, m in enumerate(df["item_id"].unique())}
        df["u_idx"] = df["user_id"].map(self.user_encoder)
        df["i_idx"] = df["item_id"].map(self.item_encoder)
        return df

    def train(self, epochs=10, batch_size=256):
        df = self.load_data()
        self.model = self._build_model(len(self.user_encoder), len(self.item_encoder))
        self.model.fit(
            [df["u_idx"].values, df["i_idx"].values],
            df["label"].values,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=2)
            ]
        )
        os.makedirs("saved_models", exist_ok=True)
        self.model.save("saved_models/ncf_model.keras")

    def recommend(self, user_id, top_n=20):
        if user_id not in self.user_encoder or self.model is None:
            return []
        u_idx     = self.user_encoder[user_id]
        all_items = list(self.item_encoder.values())
        scores    = self.model.predict(
            [np.full(len(all_items), u_idx), np.array(all_items)],
            batch_size=1024, verbose=0
        ).flatten()
        top_idxs    = np.argsort(scores)[::-1][:top_n]
        idx_to_item = {v: k for k, v in self.item_encoder.items()}
        return [(idx_to_item[i], float(scores[i])) for i in top_idxs]
