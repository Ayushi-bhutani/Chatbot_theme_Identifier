from nltk.tokenize import sent_tokenize
from typing import List, Dict

def process_document(text: str, page_number: int) -> List[Dict]:
    """Split document text into sentences with metadata"""
    sentences = sent_tokenize(text)
    return {
        "page_number": page_number,
        "full_text": text,
        "sentences": [
            {"text": s, "sentence_id": i} 
            for i, s in enumerate(sentences)
        ]
    }
