import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def run_nmf(processed_texts: List[str], n_topics: int = 5, n_top_words: int = 10, max_df: float = 0.95, min_df: int = 2) -> List[Dict[str, Any]]:
    """
    Runs Non-negative Matrix Factorization on preprocessed texts.
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    dtm = vectorizer.fit_transform(processed_texts)
    
    model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    doc_topic_dist = model.fit_transform(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = [float(topic[i]) for i in top_indices]
        
        # Find top 3 representative documents for this topic
        top_doc_indices = np.argsort(doc_topic_dist[:, topic_idx])[::-1][:3]
        representative_docs = [processed_texts[i] for i in top_doc_indices]
        
        topics.append({
            "id": topic_idx + 1,
            "words": top_words,
            "weights": weights,
            "representative_docs": representative_docs
        })
        
    return topics

