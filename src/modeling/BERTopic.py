import torch
from typing import List, Dict, Any, Optional
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Singleton for the embedding model to avoid reloading it in memory
_embedding_model_instance: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """
    Returns the singleton instance of the Gemma embedding model.
    Loads it if it hasn't been initialized yet.
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model_instance = SentenceTransformer(
            "google/embeddinggemma-300m",
            truncate_dim=256,  
            device=device
        )
    return _embedding_model_instance

def run_bertopic(processed_texts: List[str], n_topics: int = 5, n_top_words: int = 10) -> List[Dict[str, Any]]:
    """
    Runs BERTopic using Gemma embeddings and UMAP for dimensionality reduction.
    Reduce Gemma embeddings to 5 dimensions (UMAP + Truncating).
    """
    embedding_model = get_embedding_model()
    
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )
    
    model = BERTopic(
        embedding_model=embedding_model, 
        umap_model=umap_model,
        nr_topics=n_topics
    )
    
    topics, _ = model.fit_transform(processed_texts)
    
    all_topics = model.get_topics()
    formatted_topics = []
    
    for topic_id, topic_data in all_topics.items():
        if topic_id == -1: 
            continue
            
        words = [item[0] for item in topic_data[:n_top_words]]
        weights = [float(item[1]) for item in topic_data[:n_top_words]]
        
        formatted_topics.append({
            "id": int(topic_id) + 1,
            "words": words,
            "weights": weights
        })
        
    return formatted_topics
