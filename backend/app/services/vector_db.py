import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from hashlib import sha256
import time

load_dotenv()

class VectorDB:
    def __init__(self):
        # Medical-optimized embedding model
        self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        
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
            self.client.recreate_collection(  # Using recreate to ensure clean setup
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
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
        return self.model.encode(text).tolist()

    def generate_point_id(self, doc_id: str, page_num: int) -> int:
        hash_input = f"{doc_id}_{page_num}".encode("utf-8")
        return int(sha256(hash_input).hexdigest(), 16) % (10**12)

    def index_document(self, doc_id: str, pages: List[Dict]):
        """Batch indexing with medical text handling"""
        points = []
        for page in pages:
            text = page.get("text", "").strip()
            if not text or len(text) < 25:  # Skip short/noisy text
                continue

            try:
                embedding = self.embed_text(text)
                points.append(PointStruct(
                    id=self.generate_point_id(doc_id, page["page"]),
                    vector=embedding,
                    payload={
                        "doc_id": doc_id,
                        "page": page["page"],
                        "text": text,
                        "is_medical": True  # Medical document marker
                    }
                ))
            except Exception as e:
                print(f"⚠️ Embedding failed for page {page['page']}: {e}")

        if points:
            try:
                operation_result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Ensure completion
                )
                print(f"✅ Indexed {len(points)} medical text chunks from {doc_id}")
                return operation_result
            except Exception as e:
                print(f"❌ Medical document indexing failed: {e}")
                raise
        return None

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.4) -> List[Dict]:
        """Medical-optimized search with relevance filtering"""
        try:
            # Medical query expansion
            expanded_query = self._expand_medical_query(query)
            query_vector = self.embed_text(expanded_query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit*3,  # Over-fetch then filter
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Medical relevance filtering
            filtered = [
                {
                    "doc_id": hit.payload["doc_id"],
                    "page": hit.payload["page"],
                    "text": hit.payload["text"],
                    "score": hit.score,
                    "type": "medical" if hit.payload.get("is_medical") else "general"
                } 
                for hit in results 
                if hit.score > score_threshold
            ]
            
            return sorted(filtered, key=lambda x: x["score"], reverse=True)[:limit]
            
        except Exception as e:
            print(f"❌ Medical search failed: {e}")
            return []

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