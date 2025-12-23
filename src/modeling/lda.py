from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_lda(processed_texts: List[str], n_topics: int = 5, n_top_words: int = 10, max_df: float = 0.95, min_df: int = 2) -> List[Dict[str, Any]]:
    """
    Runs LDA on preprocessed texts.
    """
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    dtm = vectorizer.fit_transform(processed_texts)
    model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    model.fit(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = [float(topic[i]) for i in top_indices]
        
        topics.append({
            "id": topic_idx + 1,
            "words": top_words,
            "weights": weights
        })
        
    return topics

