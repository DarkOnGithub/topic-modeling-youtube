import torch
from typing import List, Optional, Any
from transformers import pipeline, BitsAndBytesConfig

PROMPT = """<start_of_turn>user
I have a topic represented by these keywords: {keywords_str}{docs_str}
Based on the keywords and the sample documents, what is a very short but highly descriptive title (max 3 words) for this topic?
Only provide the title, nothing else.<end_of_turn>
<start_of_turn>model
"""

_naming_model_instance: Optional[Any] = None

def get_naming_model():
    """Returns the singleton instance of Gemma 3 1b for topic naming. Quantized on 4 bits."""
    global _naming_model_instance
    if _naming_model_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Gemma 3 1b on {device}...")
        
        model_kwargs = {
            "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        _naming_model_instance = pipeline(
            "text-generation", 
            model="google/gemma-3-1b-it", 
            device_map="auto" if device == "cuda" else None,
            model_kwargs=model_kwargs
        )
    return _naming_model_instance

def generate_topic_name(keywords: List[str], documents: Optional[List[str]] = None) -> str:
    """
    Generates a concise, descriptive name (max 3 words) based on keywords and sample documents.
    Documents are optional and are used to generate a more descriptive name. 
    """
    generator = get_naming_model()
    
    keywords_str = ", ".join(keywords)
    
    docs_str = ""
    if documents:
        sample_docs = [doc[:300] + "..." if len(doc) > 300 else doc for doc in documents[:5]]
        docs_str = "\nSample documents from this topic:\n- " + "\n- ".join(sample_docs)

    prompt = PROMPT.format(keywords_str=keywords_str, docs_str=docs_str)
    
    try:
        results = generator(
            prompt, 
            max_new_tokens=50, 
            do_sample=False,
            return_full_text=False
        )
        
        generated_text = results[0]['generated_text'].strip()
        clean_name = generated_text.split("\n")[0].strip().replace("\"", "").replace("*", "")
        return clean_name if clean_name else "Unnamed Topic"
        
    except Exception as e:
        print(f"Error generating topic name: {e}")
        return "Unnamed Topic"

