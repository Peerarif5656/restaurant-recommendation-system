import pandas as pd


def train_test_split_interactions(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []

    for _, group in interactions_df.groupby('user_id'):
        group = group.sample(frac=1, random_state=random_state)

        if len(group) < 2:
            train_rows.append(group)
            continue

        test_size = max(1, int(len(group) * test_ratio))
        test_group = group.iloc[:test_size]
        train_group = group.iloc[test_size:]

        if train_group.empty:
            train_group = group.iloc[:1]
            test_group = group.iloc[1:]

        train_rows.append(train_group)
        if not test_group.empty:
            test_rows.append(test_group)

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=interactions_df.columns)
    return train_df, test_df


def evaluate_precision_recall(recommender, test_df: pd.DataFrame, k: int = 5) -> tuple[float, float]:
    if test_df.empty:
        return 0.0, 0.0

    truth_by_user = test_df.groupby('user_id')['restaurant_id'].apply(set).to_dict()
    precisions: list[float] = []
    recalls: list[float] = []

    for user_id, true_items in truth_by_user.items():
        predicted = recommender.recommend(user_id=user_id, top_k=k)
        predicted_items = {restaurant_id for restaurant_id, _ in predicted}

        if not predicted_items:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        hits = len(predicted_items.intersection(true_items))
        precisions.append(hits / k)
        recalls.append(hits / len(true_items))

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_precision, mean_recall
