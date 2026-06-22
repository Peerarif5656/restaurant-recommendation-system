from pathlib import Path
import pandas as pd


def load_data(interactions_path: Path, restaurants_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions_df = pd.read_csv(interactions_path)
    restaurants_df = pd.read_csv(restaurants_path)

    required_interaction_cols = {'user_id', 'restaurant_id', 'rating'}
    required_restaurant_cols = {'restaurant_id', 'restaurant_name', 'cuisine'}

    if not required_interaction_cols.issubset(interactions_df.columns):
        raise ValueError(f'interactions.csv must contain columns: {sorted(required_interaction_cols)}')

    if not required_restaurant_cols.issubset(restaurants_df.columns):
        raise ValueError(f'restaurants.csv must contain columns: {sorted(required_restaurant_cols)}')

    interactions_df = interactions_df.dropna(subset=['user_id', 'restaurant_id', 'rating']).copy()
    interactions_df['user_id'] = interactions_df['user_id'].astype(int)
    interactions_df['restaurant_id'] = interactions_df['restaurant_id'].astype(int)
    interactions_df['rating'] = interactions_df['rating'].astype(float)

    duplicate_mask = interactions_df.duplicated(subset=['user_id', 'restaurant_id'], keep=False)
    if duplicate_mask.any():
        interactions_df = (
            interactions_df.groupby(['user_id', 'restaurant_id'], as_index=False)['rating']
            .mean()
        )

    return interactions_df, restaurants_df


def create_user_item_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    return interactions_df.pivot_table(
        index='user_id',
        columns='restaurant_id',
        values='rating',
        aggfunc='mean',
        fill_value=0.0,
    )
