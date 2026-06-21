# Restaurant Recommendation System

This project builds a restaurant recommendation system using historical user interaction data. It demonstrates how recommendation models can support personalised restaurant discovery in food delivery and marketplace platforms.

## Project objective

The project aims to:
- generate personalised restaurant recommendations
- rank restaurants using user behaviour data
- evaluate recommendation quality with Precision\@K and Recall\@K
- present a clean, reproducible university-level machine learning workflow

## Features

- user-item interaction matrix
- collaborative filtering with cosine similarity
- top-K personalised recommendations
- popularity-based fallback for cold-start users
- evaluation with Precision\@K and Recall\@K
- modular Python code split into preprocessing, recommendation, and evaluation layers

## Project structure

```text
restaurant-recommendation-system/
│
├── data/
│   ├── interactions.csv
│   └── restaurants.csv
├── src/
│   ├── data_preprocessing.py
│   ├── recommender.py
│   └── evaluation.py
├── outputs/
│   └── recommendations.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

### interactions.csv
Contains historical user interactions with restaurants.

Columns:
- `user_id`
- `restaurant_id`
- `rating`

### restaurants.csv
Contains restaurant metadata.

Columns:
- `restaurant_id`
- `restaurant_name`
- `cuisine`

## Methodology

1. Load and validate the interaction and restaurant datasets
2. Build a user-item matrix
3. Compute user-user similarity using cosine similarity
4. Generate personalised restaurant suggestions from similar users
5. Apply a popularity-based fallback for cold-start users
6. Evaluate the system using Precision\@K and Recall\@K

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
python main.py
```

The script will:
- split the interaction data into train and test sets
- fit the recommender model
- generate recommendations for sample users
- evaluate Top-5 Precision and Recall
- save results to `outputs/recommendations.csv`

## Example output

```text
Train interactions: 46
Test interactions: 12
Top-5 Precision: 0.1333
Top-5 Recall: 0.6667
Saved recommendations to: /path/to/outputs/recommendations.csv
```

## Skills demonstrated

- Python programming
- data preprocessing
- collaborative filtering
- recommendation systems
- ranking metrics
- modular project structure
- Git and GitHub workflow

## Future improvements

- matrix factorisation
- implicit-feedback recommendations
- hybrid recommender using cuisine features
- experiment tracking with MLflow
- deployment with FastAPI or Streamlit

## Relevance to marketplace platforms

This project reflects recommendation problems faced by digital food platforms, where machine learning is used to personalise restaurant discovery, improve engagement, and support ranking quality.
