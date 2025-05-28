import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from pathlib import Path
import json

class ThemeAnalyzer:
    
    def __init__(self, json_dir: Path, min_docs: int = 2, min_word_length: int = 6):
        self.json_dir = json_dir
        self.min_docs = min_docs
        self.min_word_length = min_word_length

    def load_documents(self) -> List[Dict]:
        """Load all processed documents from JSON files"""
        documents = []
        for json_file in self.json_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                pages = json.load(f)
                full_text = " ".join(
                    s["text"] for p in pages 
                    if "sentences" in p 
                    for s in p["sentences"]
                )
                documents.append({
                    "doc_id": json_file.stem,
                    "text": full_text,
                    "pages": pages
                })
        return documents

    def analyze_themes_tfidf(self) -> List[Dict]:
        """Advanced theme detection using TF-IDF with document references"""
        documents = self.load_documents()
        
        # 1. Extract key phrases using TF-IDF
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(2, 3),  # Capture multi-word phrases
            min_df=self.min_docs  # Only consider phrases appearing in multiple docs
        )
        matrix = tfidf.fit_transform([doc['text'] for doc in documents])
        feature_names = tfidf.get_feature_names_out()
        
        # 2. Identify significant themes
        significant_phrases = set()
        for i in range(len(documents)):
            top_phrases = feature_names[matrix[i].toarray().argsort()[0][-5:]]  # Top 5 phrases per 
    
            significant_phrases.update(top_phrases)
        
        # 3. Build theme-document mapping
        themes = []
        for phrase in significant_phrases:
            relevant_docs = [
                doc['doc_id'] for doc in documents
                if phrase.lower() in doc['text'].lower()
            ]
            
            if len(relevant_docs) >= self.min_docs:
                themes.append({
                    "theme": phrase,
                    "documents": relevant_docs,
                    "snippets": self._extract_snippets(phrase, documents)
                })
        
        return themes

    def _extract_snippets(self, phrase: str, documents: List[Dict]) -> List[str]:
        """Extract context snippets around theme occurrences"""
        snippets = []
        for doc in documents:
            if phrase.lower() in doc['text'].lower():
                # Find first occurrence
                start_pos = doc['text'].lower().find(phrase.lower())
                snippet = doc['text'][max(0, start_pos-100):start_pos+len(phrase)+100]
                snippets.append(f"{doc['doc_id']}: {snippet.strip()}...")
        return snippets

# Usage Example:
if __name__ == "__main__":
    JSON_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    analyzer = ThemeAnalyzer(JSON_DIR)
    themes = analyzer.analyze_themes_tfidf()
    print(f"Found {len(themes)} themes:")
    for theme in themes:
        print(f"\nTheme: {theme['theme']}")
        print(f"Documents: {', '.join(theme['documents'])}")
        print("Sample snippet:", theme['snippets'][0][:100] + "...")
