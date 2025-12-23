import torch
import gc
import numpy as np
from typing import List, Dict, Any, Optional
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from umap import UMAP
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from src.preprocessing import BASE_STOP_WORDS_YOUTUBE

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def get_stopwords(lang_code: str = "en") -> list:
    """Returns a list of stopwords for the given language code."""
    lang_map = {
        "en": "english",
        "fr": "french",
        "es": "spanish"
    }
    nltk_lang = lang_map.get(lang_code, "english")
    try:
        stop_words = set(stopwords.words(nltk_lang))
    except Exception:
        stop_words = set(stopwords.words("english"))
    return list(stop_words.union(BASE_STOP_WORDS_YOUTUBE))

_embedding_model_instance: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """
    Returns the singleton instance of the Gemma embedding model.
    Loads it if it hasn't been initialized yet.
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_kwargs = {
            "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quantization_config
            
        model = SentenceTransformer(
            "google/embeddinggemma-300m",
            truncate_dim=256,
            device=device,
            model_kwargs=model_kwargs
        )
        model.max_seq_length = 512
        _embedding_model_instance = model
    return _embedding_model_instance

def run_bertopic(processed_texts: List[str], n_topics: int = 5, n_top_words: int = 10, lang_code: str = "en") -> List[Dict[str, Any]]:
    """
    Runs BERTopic with optimizations for large datasets and automatic localization.
    """

    embedding_model = get_embedding_model()
    
    all_stop_words = get_stopwords(lang_code)
    vectorizer_model = CountVectorizer(stop_words=all_stop_words)
    
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42,
        low_memory=True
    )
    
    min_topic_size = 50 if len(processed_texts) > 500 else 10
    
    model = BERTopic(
        embedding_model=embedding_model, 
        umap_model=umap_model,
        nr_topics=n_topics,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        verbose=True
    )

    embeddings = embedding_model.encode(
        processed_texts, 
        show_progress_bar=True, 
        batch_size=128 
    )
    
    topics, _ = model.fit_transform(processed_texts, embeddings=embeddings)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    all_topics = model.get_topics()
    representative_docs = model.get_representative_docs()
    
    formatted_topics = []
    for topic_id, topic_data in all_topics.items():
        if topic_id == -1: 
            continue
            
        words = [item[0] for item in topic_data[:n_top_words]]
        weights = [float(item[1]) for item in topic_data[:n_top_words]]
        
        docs = representative_docs.get(topic_id, [])
        
        formatted_topics.append({
            "id": int(topic_id) + 1,
            "words": words,
            "weights": weights,
            "representative_docs": docs
        })
        
    return formatted_topics
