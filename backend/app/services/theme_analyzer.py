from typing import List, Dict
from collections import defaultdict
import re
from app.services.gpt_summarizer import generate_theme_summary

class ThemeAnalyzer:
    def __init__(self):
        self.themes_cache = []  # Store identified themes for reuse

    def extract_common_themes(self, search_results: List[Dict]) -> List[Dict]:
        """
        Identify recurring themes across search results.
        Returns:
            List[Dict]: Each theme with associated documents & snippets.
        """
        word_doc_map = defaultdict(set)
        doc_snippets = {}

        for result in search_results:
            doc_id = result["doc_id"]
            text = result["text"]
            
            # Store document snippets for later summarization
            doc_snippets[doc_id] = text[:300] + "..."  # Truncate for efficiency
            
            # Extract meaningful keywords (6+ letters, excluding stopwords)
            keywords = re.findall(r'\b\w{6,}\b', text.lower())
            for word in keywords:
                word_doc_map[word].add(doc_id)

        # Filter for words appearing in multiple documents (potential themes)
        theme_candidates = {
            word: list(docs) 
            for word, docs in word_doc_map.items() 
            if len(docs) > 1
        }

        # Build theme objects with document references
        themes = []
        for theme_word, docs in theme_candidates.items():
            snippets = [f"{doc}: {doc_snippets[doc]}" for doc in docs]
            themes.append({
                "theme": theme_word,
                "documents": docs,
                "snippets": snippets
            })

        self.themes_cache = themes  # Cache for reuse
        return themes

    async def generate_synthesized_response(self, query: str, search_results: List[Dict]) -> Dict:
        """
        Generate a final response with:
        - Individual document answers (with citations)
        - Identified themes
        - GPT-summarized synthesis
        """
        themes = self.extract_common_themes(search_results)
        
        # Generate a GPT summary of the themes
        theme_summary = await generate_theme_summary(
            query=query,
            themes=[t["theme"] for t in themes],
            snippets=[s for t in themes for s in t["snippets"]]
        )

        return {
            "query": query,
            "individual_results": search_results,  # Original results with citations
            "identified_themes": themes,  # Raw themes & doc references
            "synthesized_summary": theme_summary  # GPT-generated overview
        }