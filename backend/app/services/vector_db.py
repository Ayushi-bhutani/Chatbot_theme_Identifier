import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from hashlib import sha256
import uuid
import time
import nltk
from nltk.tokenize import sent_tokenize

# Initialize NLTK
nltk.download('punkt')

load_dotenv()

class VectorDB:
    def __init__(self):
        # Medical-optimized embedding model
        self.embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        
        # Qdrant connection with timeout configuration
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30  # Increased timeout for medical documents
        )
        self.collection_name = "document_chunks"
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the vector collection with medical-optimized settings"""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "default_segment_number": 3  # Better for medical text chunks
                }
            )
            print("✅ Medical-optimized collection created")
        except Exception as e:
            print(f"Collection initialization note: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Enhanced embedding with medical text preprocessing"""
        # Basic medical text normalization
        text = text.lower().replace("diagnosis", "").replace("treatment", "")
        return self.embedding_model.encode(text).tolist()

    def generate_point_id(self, doc_id: str, page_num: int, sentence_id: int) -> int:
        """Generate unique ID for each sentence"""
        hash_input = f"{doc_id}_{page_num}_{sentence_id}".encode("utf-8")
        return int(sha256(hash_input).hexdigest(), 16) % (10**12)

    def index_document(self, doc_id: str, pages: List[Dict]):
        """Batch indexing with full document context"""
        points = []
        for page in pages:
            if "sentences" not in page:
                continue
                
            for sentence in page["sentences"]:
                text = sentence.get("text", "").strip()
                if not text or len(text) < 25:  # Skip short/noisy text
                    continue

                try:
                    vector = self.embed_text(text)
                    points.append(PointStruct(
                        id=self.generate_point_id(doc_id, page["page_number"], sentence["sentence_id"]),
                        vector=vector,
                        payload={
                            "doc_id": doc_id,
                            "page": page["page_number"],
                            "sentence_id": sentence["sentence_id"],
                            "text": text,
                            "full_text": page["full_text"],
                            "is_medical": True
                        }
                    ))
                except Exception as e:
                    print(f"⚠️ Embedding failed for page {page['page_number']}: {e}")

        if points:
            try:
                operation_result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                print(f"✅ Indexed {len(points)} medical text chunks from {doc_id}")
                return operation_result
            except Exception as e:
                print(f"❌ Medical document indexing failed: {e}")
                raise
        return None

    def _get_context(self, payload: Dict, window_size: int = 2) -> str:
        """Get surrounding sentences for context"""
        if 'full_text' not in payload or 'sentences' not in payload['full_text']:
            return payload.get('text', '')
            
        sentences = payload['full_text']['sentences']
        current_idx = payload['sentence_id']
        start = max(0, current_idx - window_size)
        end = min(len(sentences), current_idx + window_size + 1)
        return " ".join(s['text'] for s in sentences[start:end])

    def _format_citation(self, payload: Dict) -> Dict:
        """Format citation with document metadata"""
        return {
            "document": payload["doc_id"],
            "page": payload["page"],
            "sentence": payload["sentence_id"],
            "formatted": f"{payload['doc_id']}, Page {payload['page']}, Sentence {payload['sentence_id']}"
        }

    def _expand_medical_query(self, query: str) -> str:
        """Add medical synonyms to improve recall"""
        expansions = {
            "liver": "hepatic liver",
            "disease": "disorder condition syndrome",
            "patient": "case subject individual"
        }
        for term, synonyms in expansions.items():
            if term in query.lower():
                query += " " + synonyms
        return query

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.4) -> List[Dict]:
        """Medical-optimized search with context and citations"""
        try:
            # Medical query expansion
            expanded_query = self._expand_medical_query(query)
            query_vector = self.embed_text(expanded_query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                top=top_k*3,  # Over-fetch then filter
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Process results with context and citations
            processed_results = []
            for hit in results:
                if hit.score > score_threshold:
                    processed_results.append({
                        "doc_id": hit.payload["doc_id"],
                        "page": hit.payload["page"],
                        "sentence_id": hit.payload.get("sentence_id", -1),
                        "text": hit.payload["text"],
                        "context": self._get_context(hit.payload),
                        "score": hit.score,
                        "type": "medical" if hit.payload.get("is_medical") else "general",
                        "citation": {
        "doc_id": "DOC123",
        "page": 5,
        "paragraph": 2,
        "sentence": 12,
        "text_excerpt": "The study found...",
        "surrounding_context": "Previous research... [text]... Subsequent work..."
    }
                        
                    
                    })
            
            return sorted(processed_results, key=lambda x: x["score"], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"❌ Medical search failed: {e}")
            return []