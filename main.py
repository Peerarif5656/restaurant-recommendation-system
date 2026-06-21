from pathlib import Path
import pandas as pd

from src.data_preprocessing import load_data, create_user_item_matrix
from src.recommender import RestaurantRecommender
from src.evaluation import train_test_split_interactions, evaluate_precision_recall


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / 'data'

    interactions_df, restaurants_df = load_data(
        data_path / 'interactions.csv',
        data_path / 'restaurants.csv'
    )

    train_df, test_df = train_test_split_interactions(
        interactions_df,
        test_ratio=0.25,
        random_state=42,
    )

    user_item_matrix = create_user_item_matrix(train_df)

    recommender = RestaurantRecommender()
    recommender.fit(train_df, user_item_matrix)

    sample_user_ids = sorted(train_df['user_id'].unique())[:3]
    recommendation_rows: list[dict] = []

    for user_id in sample_user_ids:
        recommendations = recommender.recommend(user_id=user_id, top_k=5)
        for rank, (restaurant_id, score) in enumerate(recommendations, start=1):
            restaurant_row = restaurants_df.loc[
                restaurants_df['restaurant_id'] == restaurant_id,
                ['restaurant_name', 'cuisine']
            ].iloc[0]
            recommendation_rows.append(
                {
                    'user_id': user_id,
                    'rank': rank,
                    'restaurant_id': restaurant_id,
                    'restaurant_name': restaurant_row['restaurant_name'],
                    'cuisine': restaurant_row['cuisine'],
                    'score': round(float(score), 4),
                }
            )

    outputs_path = base_path / 'outputs'
    outputs_path.mkdir(exist_ok=True)
    output_file = outputs_path / 'recommendations.csv'
    pd.DataFrame(recommendation_rows).to_csv(output_file, index=False)

    precision_k, recall_k = evaluate_precision_recall(
        recommender=recommender,
        test_df=test_df,
        k=5,
    )

    print(f'Train interactions: {len(train_df)}')
    print(f'Test interactions: {len(test_df)}')
    print(f'Top-5 Precision: {precision_k:.4f}')
    print(f'Top-5 Recall: {recall_k:.4f}')
    print(f'Saved recommendations to: {output_file}')


if __name__ == '__main__':
    main()
