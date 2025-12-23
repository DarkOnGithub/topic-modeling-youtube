from typing import Dict, Any
from src.preprocessing import preprocess_corpus
from src.utils import load_comments_from_folder
from .lda import run_lda
from .nmf import run_nmf
from .BERTopic import run_bertopic
from src.modeling_methods import ModelingMethod

def run_topic_modeling(
    channel_folder: str, 
    method: ModelingMethod = ModelingMethod.LDA, 
    n_topics: int = 5, 
    n_top_words: int = 10
) -> Dict[str, Any]:
    """
    Factory that redirects to the correct modeling method (LDA or NMF).
    """
    raw_texts = load_comments_from_folder(channel_folder)
    if not raw_texts:
        return {"error": "No comments found."}

    processed_texts = preprocess_corpus(raw_texts, method)
    if not processed_texts:
        return {"error": "No valid text after preprocessing."}

    if method == ModelingMethod.LDA:
        topics = run_lda(processed_texts, n_topics, n_top_words)
    elif method == ModelingMethod.NMF:
        topics = run_nmf(processed_texts, n_topics, n_top_words)
    elif method == ModelingMethod.BERTOPIC:
        topics = run_bertopic(processed_texts, n_topics, n_top_words)
    else:
        return {"error": f"Method '{method}' not recognized."}

    return {
        "channel": channel_folder,
        "method": method.value.upper(),
        "total_comments": len(processed_texts),
        "topics": topics
    }
