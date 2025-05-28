import time
from faker import Faker
from app.services.vector_db import VectorDB  # Import your vector DB class
import os
import sys
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.vector_db import VectorDB
# Initialize dependencies
load_dotenv()
fake = Faker()
vector_db = VectorDB()  # Initialize your vector DB client

def test_document_scale(num_docs=100):
    """Performance test for large document processing"""
    print(f"\n🔍 Running scale test with {num_docs} documents...")
    
    start = time.time()
    for i in range(num_docs):
        doc = {
            "text": fake.text(2000),
            "metadata": {
                "source": f"doc_{i}.pdf",
                "pages": [{
                    "page_number": 1,
                    "sentences": [
                        {"text": fake.sentence(), "sentence_id": j}
                        for j in range(20)  # 20 sentences per doc
                    ]
                }]
            }
        }
        vector_db.index_document(f"doc_{i}", doc["metadata"]["pages"])
    
    duration = time.time() - start
    print(f"✅ Processed {num_docs} docs in {duration:.2f} seconds")
    print(f"⏱️ Average time per doc: {(duration/num_docs):.3f}s")
    
    return duration

if __name__ == "__main__":
    test_document_scale()
