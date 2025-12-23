import re
import string
import spacy
from src.modeling_methods import ModelingMethod
from typing import List
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
BASE_STOP_WORDS_YOUTUBE = [
    # YouTube Platform Terms
    "video", "videos", "channel", "channels", "subscribe", "subscriber", "subscribers", 
    "sub", "subs", "comment", "comments", "watch", "watching", "watcher", "view", "views", 
    "viewer", "viewers", "click", "link", "links", "post", "posts", "upload", "uploads", 
    "content", "creator", "creators", "youtube", "yt", "algorithm", "bell", "notification", 
    "notifications", "share", "shared", "vlog", "vlogger", "vlogs", "playlist",
    
    # Common Filler & Interaction
    "thanks", "thank", "merci", "hey", "hello", "hi", "guys", "everyone", "love", 
    "like", "good", "great", "nice", "awesome", "amazing", "really", "actually", 
    "probably", "maybe", "also", "another", "thing", "things", "much", "many", 
    "lot", "lots", "everyone", "anyone", "someone", "please", "pls", "svp",
    
    # Time-related (Frequent in comments)
    "time", "times", "minute", "minutes", "hour", "hours", "day", "days", 
    "week", "weeks", "year", "years", "month", "months", "first", "second", 
    "early", "late", "ago", "today", "yesterday", "tomorrow", "soon", "now", "ever",
]

def _clean_text(text: str) -> str:
    """
    Clean text by removing URLs, mentions, punctuation, numbers and extra spaces.
    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _preprocessing(text: str) -> str:
    """
    Tokenizer, Stopwords removal, and Lemmatization.
    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = _clean_text(text)
    if not text:
        return ""
        
    doc = nlp(text)
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop 
        and token.text not in BASE_STOP_WORDS_YOUTUBE 
        and len(token.text) > 2 
        and not token.is_punct
    ]
    return " ".join(tokens)


def preprocess_text(text: str, method: ModelingMethod) -> str:
    """
    Preprocess text by cleaning and advanced preprocessing.
    Args:
        text (str): The text to preprocess.
        method (ModelingMethod): The method to use for preprocessing.

    Returns:
        str: The preprocessed text.
    """
    if method == ModelingMethod.LDA or method == ModelingMethod.NMF:
        return _preprocessing(text)
    elif method == ModelingMethod.BERTOPIC:
        return _clean_text(text)

def preprocess_corpus(corpus: List[str], method: ModelingMethod) -> List[str]:
    """
    Preprocess a corpus of texts.
    Args:
        corpus (List[str]): The corpus of texts to preprocess.
        method (ModelingMethod): The method to use for preprocessing.

    Returns:
        List[str]: The preprocessed corpus.
    """
    processed_texts = [preprocess_text(text, method) for text in corpus]
    return [text for text in processed_texts if text.strip()]