import re
import string
import spacy
import langid
from src.modeling_methods import ModelingMethod
from typing import List, Dict, Optional

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
}

_nlp_cache: Dict[str, spacy.language.Language] = {}

def get_nlp(lang_code: str = "en") -> spacy.language.Language:
    """
    Returns a cached spaCy model for the given language code.
    Falls back to English if the model is not found or not installed.
    """
    global _nlp_cache
    
    model_name = SPACY_MODELS.get(lang_code, SPACY_MODELS["en"])
    
    if model_name not in _nlp_cache:
        try:
            print(f"Loading spaCy model: {model_name}...")
            _nlp_cache[model_name] = spacy.load(model_name, disable=["parser", "ner"])
        except OSError:
            print(f"Warning: spaCy model {model_name} not found. Falling back to English.")
            if "en_core_web_sm" not in _nlp_cache:
                _nlp_cache["en_core_web_sm"] = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            return _nlp_cache["en_core_web_sm"]
            
    return _nlp_cache[model_name]

def detect_language(text: str) -> str:
    """Detects the language of a text."""
    if not text or len(text) < 5:
        return "en"
    lang, _ = langid.classify(text)
    return lang

nlp = get_nlp("en")
    
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

def _preprocessing(text: str, nlp_model: Optional[spacy.language.Language] = None) -> str:
    """
    Tokenizer, Stopwords removal, and Lemmatization.
    """
    text = _clean_text(text)
    if not text:
        return ""
        
    # Use provided model or fall back to global nlp
    model = nlp_model if nlp_model else nlp
    doc = model(text)
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop 
        and token.text not in BASE_STOP_WORDS_YOUTUBE 
        and len(token.text) > 2 
        and not token.is_punct
    ]
    return " ".join(tokens)


def preprocess_text(text: str, method: ModelingMethod, nlp_model: Optional[spacy.language.Language] = None) -> str:
    """
    Preprocess text by cleaning and advanced preprocessing.
    """
    if method == ModelingMethod.LDA or method == ModelingMethod.NMF:
        return _preprocessing(text, nlp_model)
    elif method == ModelingMethod.BERTOPIC:
        return _clean_text(text)

def preprocess_corpus(corpus: List[str], method: ModelingMethod) -> tuple[List[str], str]:
    """
    Preprocess a corpus of texts with automatic language detection.
    Returns (processed_texts, lang_code).
    """
    if not corpus:
        return [], "en"

    # Automatic Localization: Detect dominant language from a sample of the corpus
    # We sample up to 20 comments to decide the language for the whole batch
    sample_text = " ".join(corpus[:20])
    lang_code = detect_language(sample_text)
    nlp_model = get_nlp(lang_code)
    
    print(f"Detected dominant language: {lang_code}. Using {SPACY_MODELS.get(lang_code, 'en_core_web_sm')} for processing.")

    processed_texts = [preprocess_text(text, method, nlp_model) for text in corpus]
    valid_texts = [text for text in processed_texts if text.strip()]
    
    return valid_texts, lang_code
