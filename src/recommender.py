import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class RestaurantRecommender:
    def __init__(self) -> None:
        self.interactions_df: pd.DataFrame | None = None
        self.user_item_matrix: pd.DataFrame | None = None
        self.user_similarity: pd.DataFrame | None = None
        self.popularity_scores: pd.Series | None = None

    def fit(self, interactions_df: pd.DataFrame, user_item_matrix: pd.DataFrame) -> None:
        self.interactions_df = interactions_df.copy()
        self.user_item_matrix = user_item_matrix.copy()

        similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        self.popularity_scores = (
            interactions_df.groupby('restaurant_id')['rating']
            .mean()
            .sort_values(ascending=False)
        )

    def recommend(self, user_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        if self.interactions_df is None or self.user_item_matrix is None or self.user_similarity is None:
            raise ValueError('Model has not been fitted yet.')

        if user_id not in self.user_item_matrix.index:
            return self._popular_recommendations(top_k)

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_restaurants = set(user_ratings[user_ratings > 0].index.tolist())

        similar_users = self.user_similarity.loc[user_id].drop(index=user_id).sort_values(ascending=False)
        recommendation_scores: dict[int, float] = {}

        for neighbour_id, similarity_score in similar_users.items():
            if similarity_score <= 0:
                continue

            neighbour_ratings = self.user_item_matrix.loc[neighbour_id]
            for restaurant_id, rating in neighbour_ratings.items():
                if rating <= 0 or restaurant_id in rated_restaurants:
                    continue
                recommendation_scores[restaurant_id] = recommendation_scores.get(restaurant_id, 0.0) + (
                    float(similarity_score) * float(rating)
                )

        if not recommendation_scores:
            return self._popular_recommendations(top_k, exclude_ids=rated_restaurants)

        ranked = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _popular_recommendations(
        self,
        top_k: int,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self.popularity_scores is None:
            raise ValueError('Popularity scores are not available. Fit the model first.')

        exclude_ids = exclude_ids or set()
        results: list[tuple[int, float]] = []

        for restaurant_id, score in self.popularity_scores.items():
            if int(restaurant_id) in exclude_ids:
                continue
            results.append((int(restaurant_id), float(score)))
            if len(results) == top_k:
                break

        return results
