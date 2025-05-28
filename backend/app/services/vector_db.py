from app.utils.imports import *
from typing import Tuple
import logging

class VectorDB:
    """Medical document vector database service with Qdrant backend.
    
    Features:
    - Sentence-level embedding storage
    - Paragraph-aware citations
    - Medical text optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=30
            )
            self._init_collection()
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _init_collection(self) -> None:
        """Initialize Qdrant collection with medical-optimized settings."""
        self.client.recreate_collection(
            collection_name="medical_docs",
            vectors_config=VectorParams(
                size=self.model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
            optimizers_config={"default_segment_number": 3}
        )

    def _preprocess_medical_text(self, text: str) -> str:
        """Clean medical text for better embeddings."""
        text = text.lower().strip()
        # Remove common noisy patterns
        patterns = [
            r"\bdoi:.+?\b", r"\bPMID:.+?\b", 
            r"\[.*?\]", r"\(.*?\)"
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return " ".join(text.split())

    def generate_doc_fingerprint(self, doc_id: str, page: int, sentence: int) -> str:
        """Create unique ID using document metadata."""
        return sha256(f"{doc_id}_{page}_{sentence}".encode()).hexdigest()[:12]

    def index_document(self, doc_id: str, pages: List[Dict]) -> Tuple[int, List[str]]:
        """Index document with error handling and validation."""
        successes, errors = [], []
        
        for page in pages:
            if not page.get("sentences"):
                continue
                
            for sentence in page["sentences"]:
                try:
                    text = self._preprocess_medical_text(sentence["text"])
                    if len(text) < 25:  # Skip short sentences
                        continue
                        
                    embedding = self.model.encode(text)
                    point_id = self.generate_doc_fingerprint(
                        doc_id, 
                        page["page_number"], 
                        sentence["sentence_id"]
                    )
                    
                    self.client.upsert(
                        collection_name="medical_docs",
                        points=[PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "doc_id": doc_id,
                                "page": page["page_number"],
                                "sentence_id": sentence["sentence_id"],
                                "text": text,
                                "full_text": page["full_text"]
                            }
                        )]
                    )
                    successes.append(point_id)
                except Exception as e:
                    errors.append(f"Doc {doc_id} page {page['page_number']}: {str(e)}")
        
        return len(successes), errors

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Medical-optimized semantic search with citations."""
        try:
            processed_query = self._preprocess_medical_text(query)
            vector = self.model.encode(processed_query)
            
            results = self.client.search(
                collection_name="medical_docs",
                query_vector=vector,
                limit=limit,
                with_payload=True
            )
            
            return [{
                "doc_id": hit.payload["doc_id"],
                "page": hit.payload["page"],
                "sentence_id": hit.payload["sentence_id"],
                "text": hit.payload["text"],
                "score": hit.score,
                "citation": self._format_citation(hit.payload),
                "context": self._get_context(hit.payload)
            } for hit in results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []

    def _format_citation(self, payload: Dict) -> Dict:
        """Generate academic-style citation."""
        return {
            "source": payload["doc_id"],
            "location": {
                "page": payload["page"],
                "paragraph": payload["text"].count("\n\n") + 1,
                "sentence": payload["sentence_id"]
            },
            "preview": f"{payload['text'][:100]}..."
        }

    def _get_context(self, payload: Dict, window: int = 2) -> str:
        """Extract surrounding sentences for context."""
        sentences = sent_tokenize(payload["full_text"])
        current_idx = payload["sentence_id"]
        start = max(0, current_idx - window)
        end = min(len(sentences), current_idx + window + 1)
        return " [...] ".join(sentences[start:end])
