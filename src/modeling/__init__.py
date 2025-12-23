from typing import Dict, Any
from src.preprocessing import preprocess_corpus
from src.utils import load_comments_from_folder
from .lda import run_lda
from .nmf import run_nmf
from .BERTopic import run_bertopic
from src.modeling_methods import ModelingMethod
from src.topic_naming import generate_topic_name
import numpy as np


MAX_COMMENTS = 40000

def run_topic_modeling(
    channel_folder: str, 
    method: ModelingMethod = ModelingMethod.LDA, 
    n_topics: int = 5, 
    n_top_words: int = 10
) -> Dict[str, Any]:
    """
    Factory that redirects to the correct modeling method (LDA, NMF, or BERTopic)
    and applies LLM-based topic naming.
    """
    raw_texts = load_comments_from_folder(channel_folder)
    if not raw_texts:
        return {"error": "No comments found."}

    processed_texts, lang_code = preprocess_corpus(raw_texts, method)
    if not processed_texts:
        return {"error": "No valid text after preprocessing."}
    
    if len(processed_texts) > MAX_COMMENTS:
        indices = np.random.choice(len(processed_texts), MAX_COMMENTS, replace=False)
        processed_texts = [processed_texts[i] for i in indices]
        print(f"Sampled {len(processed_texts)} comments from {len(raw_texts)} total comments.")
        
    if method == ModelingMethod.LDA:
        topics = run_lda(processed_texts, n_topics, n_top_words)
    elif method == ModelingMethod.NMF:
        topics = run_nmf(processed_texts, n_topics, n_top_words)
    elif method == ModelingMethod.BERTOPIC:
        topics = run_bertopic(processed_texts, n_topics, n_top_words, lang_code=lang_code)
    else:
        return {"error": f"Method '{method}' not recognized."}
    
    for topic in topics:
        documents = topic.get("representative_docs")[:5]
        if "words" in topic and topic["words"]:
            topic["name"] = generate_topic_name(
                topic["words"], 
                documents=documents,
                lang_code=lang_code
            )

    return {
        "channel": channel_folder,
        "method": method.value.upper(),
        "total_comments": len(processed_texts),
        "topics": topics
    }
