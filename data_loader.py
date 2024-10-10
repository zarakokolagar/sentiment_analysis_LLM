# Imports and Configuration
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,util
from dataclasses import dataclass
from textblob import TextBlob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from config import DATA_DIR

@dataclass
class DatasetConfig:
    dataset_name: str = "ajaykarthick/imdb-movie-reviews"
    split: str = 'train'
    label_0_samples: int = 3000
    label_1_samples: int = 3000
    shuffle_seed: int = 42
    embedding_model: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32  # Batch size for embedding calculation

@dataclass
class SimilarityConfig:
    num_least_similar: int = 50  # Number of least similar samples to select per class

def load_and_preprocess_data(config: DatasetConfig) -> pd.DataFrame:
    """Load and preprocess the dataset by selecting samples from each class."""
    dataset = load_dataset(config.dataset_name, split=config.split)
    dataset = dataset.shuffle(seed=config.shuffle_seed)

    # Select the required samples by filtering and concatenating
    label_0_samples = [i for i, x in enumerate(dataset) if x['label'] == 0][:config.label_0_samples]
    label_1_samples = [i for i, x in enumerate(dataset) if x['label'] == 1][:config.label_1_samples]
    selected_indices = label_0_samples + label_1_samples
    subset = dataset.select(selected_indices)

    # Convert to DataFrame and map labels
    df = pd.DataFrame(subset)
    df['label'] = df['label'].map({0: 'positive', 1: 'negative'})

    return df

def predict_sentiment_textblob(df: pd.DataFrame) -> pd.DataFrame:
    """Predict sentiment using TextBlob and add the predicted labels to the DataFrame."""
    reviews = df['review'].tolist()
    textblob_predictions = ['positive' if TextBlob(review).sentiment.polarity > 0 else 'negative' for review in reviews]
    df['textblob_predicted_label'] = textblob_predictions
    return df

def sample_reviews(df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """Sample reviews ensuring at least n_samples per class."""
    matching_reviews = df[df['label'] == df['textblob_predicted_label']]
    sampled_positive = matching_reviews[matching_reviews['label'] == 'positive'].sample(n=n_samples, random_state=42)
    sampled_negative = matching_reviews[matching_reviews['label'] == 'negative'].sample(n=n_samples, random_state=42)
    final_sampled_reviews = pd.concat([sampled_positive, sampled_negative]).reset_index(drop=True)
    return final_sampled_reviews

def get_embeddings(reviews: list, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for a list of reviews using SentenceTransformer."""
    return model.encode(reviews, convert_to_tensor=True).cpu().numpy()

def get_least_similar_reviews(df: pd.DataFrame, embeddings: np.ndarray, sim_config: SimilarityConfig) -> pd.DataFrame:
    """Find the least similar reviews within each class using cosine similarity."""
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    least_similar_indices = []

    for label in ['positive', 'negative']:
        class_indices = df.index[df['label'] == label].tolist()
        class_similarity = similarity_matrix[class_indices][:, class_indices]
        avg_similarity = class_similarity.mean(dim=1).cpu().numpy()
        least_similar_indices += [class_indices[i] for i in np.argsort(avg_similarity)[:sim_config.num_least_similar]]

    least_similar_reviews = df.loc[least_similar_indices].reset_index(drop=True)
    return least_similar_reviews

# Main execution
dataset_config = DatasetConfig()
df = load_and_preprocess_data(dataset_config)
df = predict_sentiment_textblob(df)
final_sampled_reviews = sample_reviews(df, n_samples=500)

# Load the embedding model for generating embeddings
embedding_model = SentenceTransformer(dataset_config.embedding_model)

# Generate embeddings for the final sampled reviews
embeddings = get_embeddings(final_sampled_reviews['review'].tolist(), embedding_model)

# Get least similar reviews
similarity_config = SimilarityConfig(num_least_similar=50)
least_similar_reviews = get_least_similar_reviews(final_sampled_reviews, embeddings, similarity_config)


# Save the data
least_similar_reviews.to_csv(os.path.join(DATA_DIR,'sampled_dataset.tsv'), sep='\t', index=False)