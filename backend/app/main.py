import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import shutil
import pytesseract
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF
import json
import re
import traceback
import uuid
import magic  # python-magic package
import nltk
from datetime import datetime
from app.services.vector_db import VectorDB
from app.services.gpt_summarizer import generate_theme_summary
import nltk
from fastapi import APIRouter
from app.services.vector_db import VectorDB
from app.services.theme_analyzer import ThemeAnalyzer
from app.schemas import QueryRequest
from collections import defaultdict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')  # This downloads the tokenizer models
from nltk.tokenize import sent_tokenize
from app.services.theme_analyzer import ThemeAnalyzer
from app.static.theme_visualizer import router as viz_router
from fastapi.responses import StreamingResponse
from io import BytesIO
from app.services.report_generator import ReportGenerator
from app.services.theme_analyzer import ThemeAnalyzer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔧 Replace all `print()` with:
logger.info("message") or logger.error("message")

# Load environment variables
import os
from dotenv import load_dotenv

# ✅ Load the .env from backend folder
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# ✅ Validate key is loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
print("Loaded OPENAI_API_KEY =", openai_api_key)

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set!")

# ✅ Then assign it to OpenAI
import openai
openai.api_key = openai_api_key

# Initialize services
vector_db = VectorDB()
app = FastAPI()
# theme_analyzer = ThemeAnalyzer()
theme_analyzer = ThemeAnalyzer(json_dir="../data/extracted_json")
app.include_router(viz_router, prefix="/api")
# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data"
PDF_DIR = UPLOAD_DIR / "uploaded"
TEXT_DIR = UPLOAD_DIR / "extracted"
JSON_DIR = UPLOAD_DIR / "extracted_json"

for folder in [UPLOAD_DIR, PDF_DIR, TEXT_DIR, JSON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Pydantic Models ---
class ThemeSummaryRequest(BaseModel):
    theme: str
    documents: List[str]
    summary_snippets: List[str]

class ThemeResponse(BaseModel):
    theme: str
    gpt_summary: str

class QueryRequest(BaseModel):
    text: str
    filter_by: Optional[dict] = None

# --- Document Processing Functions ---
def process_document(text: str, page_number: int) -> List[Dict]:
    """Split document text into sentences with metadata"""
    try:
        sentences = sent_tokenize(text)
        return {
            "page_number": page_number,
            "full_text": text,
            "sentences": [
                {"text": s, "sentence_id": i} 
                for i, s in enumerate(sentences)
            ]
        }
    except Exception as e:
        print(f"Sentence tokenization error: {e}")
        # Fallback to simple split if tokenization fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return {
            "page_number": page_number,
            "full_text": text,
            "sentences": [
                {"text": s, "sentence_id": i} 
                for i, s in enumerate(sentences)
            ],
            "warning": "Used simple sentence splitting"
        }

def extract_text_from_pdf(pdf_path: Path) -> List[dict]:
    """Enhanced PDF text extraction with metadata and sentence tokenization"""
    text_data = []
    metadata = {
        "filename": pdf_path.name,
        "upload_date": datetime.now().isoformat(),
        "page_count": 0,
        "is_scanned": False
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata["page_count"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                content = page.extract_text()
                if not content:
                    content = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if not content:
                        metadata["is_scanned"] = True
                        raise ValueError("Possible scanned document")
                
                page_data = process_document(content.strip(), i)
                text_data.append(page_data)
                
    except Exception as e:
        print(f"[INFO] Falling back to OCR processing: {str(e)}")
        text_data = ocr_scanned_pdf(pdf_path)
        metadata["is_scanned"] = True
    
    if text_data:
        text_data[0]["metadata"] = metadata
    
    return text_data

def ocr_scanned_pdf(pdf_path: Path) -> List[dict]:
    """Improved OCR processing with better image handling"""
    text_by_page = []
    doc = fitz.open(pdf_path)
    
    for i, page in enumerate(doc, start=1):
        try:
            pix = page.get_pixmap(dpi=300)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
            
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            page_data = process_document(text.strip(), i)
            page_data["ocr_processed"] = True
            text_by_page.append(page_data)
        except Exception as e:
            print(f"Error processing page {i}: {str(e)}")
            text_by_page.append({
                "page_number": i,
                "sentences": [],
                "error": str(e)
            })
    
    return text_by_page

# --- API Endpoints ---
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handle PDF uploads with validation and processing"""
    file_content = await file.read(2048)
    await file.seek(0)

    mime_type = magic.from_buffer(file_content, mime=True)
    if mime_type != 'application/pdf':
        raise HTTPException(400, "Only PDF files are accepted")

    max_size = 50 * 1024 * 1024  # 50MB
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)

    if file_size > max_size:
        raise HTTPException(400, "File too large. Max 50MB allowed")

    pdf_path = PDF_DIR / file.filename
    if pdf_path.exists():
        raise HTTPException(400, "File with this name already exists")

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        with fitz.open(pdf_path) as doc:
            if not doc.is_pdf:
                os.remove(pdf_path)
                raise HTTPException(400, "Invalid PDF structure")
    except:
        os.remove(pdf_path)
        raise HTTPException(400, "Invalid PDF file")

    try:
        extracted_data = extract_text_from_pdf(pdf_path)

        # Fallback to OCR if document appears scanned
        if extracted_data[0].get('metadata', {}).get('is_scanned'):
            extracted_data = apply_enhanced_ocr(pdf_path)

        json_path = JSON_DIR / f"{file.filename}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(extracted_data, jf, ensure_ascii=False, indent=2)

        try:
            vector_db.index_document(file.filename, extracted_data)
        except Exception as vec_err:
            logger.error(f"Vector DB indexing error: {vec_err}")

        return {
            "filename": file.filename,
            "message": "Uploaded, text extracted, and indexed successfully.",
            "metadata": extracted_data[0].get("metadata", {}) if extracted_data else {},
            "vector_status": "indexed"
        }

    except Exception as e:
        if pdf_path.exists():
            os.remove(pdf_path)
        json_path = JSON_DIR / f"{file.filename}.json"
        if json_path.exists():
            os.remove(json_path)
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")

@app.get("/documents/")
def list_documents():
    documents = []
    for json_file in JSON_DIR.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        metadata = data[0].get("metadata", {}) if data else {}
        documents.append({
            "filename": json_file.stem,
            "page_count": metadata.get("page_count", len(data)),
            "upload_date": metadata.get("upload_date", "N/A"),
            "is_scanned": metadata.get("is_scanned", False),
            "size_kb": os.path.getsize(PDF_DIR / f"{json_file.stem}.pdf") // 1024 if (PDF_DIR / f"{json_file.stem}.pdf").exists() else 0
        })
    return {"documents": documents}

@app.post("/query/")
async def query_documents(
    question: str = Query(..., min_length=3),
    semantic: bool = Query(True),
    keyword: bool = Query(False),
    limit: int = Query(5),
    analyze_themes: bool = Query(True, description="Enable theme analysis")
):
    """
    Search documents with semantic and keyword options.
    Returns results with citations and optional theme analysis.
    """
    # Initialize variables
    results = []
    
    theme_analyzer = ThemeAnalyzer(json_dir="../data/extracted_json")
    
    # Semantic Search (Vector DB)
    if semantic:
        try:
            vector_results = vector_db.search(question, limit=limit*3)  # Over-fetch for theme analysis
            for res in vector_results:
                results.append({
                    "document": res["doc_id"],
                    "page": res["page"],
                    "sentence": res["sentence_id"],
                    "excerpt": highlight_query(res["text"], question),
                    "text": res["text"],  # Keep full text for theme analysis
                    "score": float(res["score"]),
                    "type": "semantic",
                    "citation": f"{res['doc_id']}, Page {res['page']}, Sentence {res['sentence_id']}"
                })
        except Exception as e:
            print(f"Vector search error: {e}")
            if not keyword:
                raise HTTPException(500, "Semantic search temporarily unavailable")

    # Keyword Search (Fallback or hybrid mode)
    if keyword:
        for file in JSON_DIR.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                pages = json.load(f)
                for page in pages:
                    if "sentences" in page:
                        for sent in page["sentences"]:
                            if question.lower() in sent["text"].lower():
                                results.append({
                                    "document": file.stem,
                                    "page": page["page_number"],
                                    "sentence": sent["sentence_id"],
                                    "excerpt": highlight_query(sent["text"], question),
                                    "text": sent["text"],  # Keep full text for theme analysis
                                    "score": 0.5,
                                    "type": "keyword",
                                    "citation": f"{file.stem}, Page {page['page_number']}, Sentence {sent['sentence_id']}"
                                })

    # Sort and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    final_results = results[:limit]
    
    # Prepare response
    response = {
        "query": question,
        "results": final_results,
        "count": len(final_results)
    }

    # Add theme analysis if enabled
    if analyze_themes and len(results) > 1:
        try:
            theme_response = await theme_analyzer.generate_synthesized_response(
                query=question,
                search_results=results  # Use all results (not just limited ones) for better theme detection
            )
            response.update({
                "themes": theme_response["identified_themes"],
                "synthesis": theme_response["synthesized_summary"]
            })
        except Exception as e:
            print(f"Theme analysis failed: {e}")
            response["theme_analysis_error"] = "Could not generate themes"

    return response

# @app.get("/synthesize/")


@app.post("/summarize_theme/", response_model=ThemeResponse)
async def summarize_theme(request: ThemeSummaryRequest):
    """Generate GPT summary for a theme"""
    if not request.summary_snippets or not request.theme:
        raise HTTPException(422, "Theme and snippets must be provided")
    
    summary = generate_theme_summary(request.theme, request.summary_snippets)
    return {
    "theme": request.theme,
    "documents": request.documents,
    "gpt_summary": summary
}



# --- Utility Functions ---
def highlight_query(text: str, query: str, snippet_length: int = 300) -> str:
    """Highlight query terms in text snippet"""
    text_lower = text.lower()
    query_lower = query.lower()
    
    start_pos = text_lower.find(query_lower)
    if start_pos == -1:
        return text[:snippet_length] + "..." if len(text) > snippet_length else text
    
    snippet_start = max(0, start_pos - snippet_length//2)
    snippet_end = min(len(text), start_pos + len(query) + snippet_length//2)
    snippet = text[snippet_start:snippet_end]
    
    for term in query.split():
        if len(term) > 3:
            snippet = re.sub(f"({term})", r"[**\1**]", snippet, flags=re.IGNORECASE)
    
    return ("..." if snippet_start > 0 else "") + snippet + ("..." if snippet_end < len(text) else "")

# --- Debug Endpoints ---
@app.get("/debug/{filename}")
def debug_document(filename: str):
    """Debug endpoint for document inspection"""
    json_path = JSON_DIR / f"{filename}.json"
    if not json_path.exists():
        return {"error": "File not processed"}
    
    with open(json_path, "r") as f:
        content = json.load(f)
    
    vector_results = vector_db.search("sample query", limit=1)
    return {
        "text_extracted": bool(content),
        "vector_db_has_doc": any(r["doc_id"] == filename for r in vector_results),
        "first_page_text": content[0]["sentences"][0]["text"][:200] if content and "sentences" in content[0] else None
    }

@app.post("/reindex/")
def reindex_all():
    """Reindex all documents in the vector database"""
    for pdf in PDF_DIR.glob("*.pdf"):
        extracted = extract_text_from_pdf(pdf)
        vector_db.index_document(pdf.name, extracted)
    return {"status": "reindexed"}
from app.services.report_generator import ReportGenerator

@app.get("/api/report")
async def generate_report():
    analyzer = ThemeAnalyzer(JSON_DIR)
    documents = [...]  # Load your documents
    themes = analyzer.extract_common_themes(documents)
    
    reporter = ReportGenerator()
    report = await reporter.generate_theme_report(themes, documents)
    
    return {
        "report": report,
        "visualization_url": f"/api/visualize?search={report['statistics']['most_common_theme']}"
    }
from pdf2image import convert_from_path
import pytesseract

def apply_enhanced_ocr(pdf_path: Path) -> List[Dict]:
    """Process scanned PDFs with OCR and return structured sentence data"""
    images = convert_from_path(pdf_path, dpi=300)
    structured_pages = []

    for i, image in enumerate(images):
        try:
            # Enhance image and extract raw text
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(image, config=custom_config)
            sentences = sent_tokenize(raw_text)

            structured_pages.append({
                "page_number": i + 1,
                "full_text": raw_text,
                "sentences": [
                    {"text": s.strip(), "sentence_id": idx}
                    for idx, s in enumerate(sentences) if s.strip()
                ]
            })
        except Exception as e:
            print(f"OCR failed for page {i+1}: {e}")
            structured_pages.append({
                "page_number": i + 1,
                "full_text": "",
                "sentences": [],
                "error": str(e)
            })

    return structured_pages

from collections import defaultdict
import networkx as nx
import plotly.graph_objects as go
def extract_key_themes(text: str, n_themes: int = 5) -> List[str]:
    """Extract key themes from text using TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=n_themes,
        stop_words='english',
        ngram_range=(1, 2)  # Include single words and bigrams
    )
    
    # Process text and get top weighted terms
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(
        zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0]),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [term for term, score in sorted_items[:n_themes]]
@app.get("/theme_network")
def generate_theme_network(min_docs: int = 3):
    """Generate interactive network graph of themes"""
    # 1. Extract themes and document relationships
    theme_docs = defaultdict(list)
    for json_file in JSON_DIR.glob("*.json"):
        with open(json_file, encoding='utf-8') as f:
            content = f.read()
            print("File starts with:", content[:100])  # Preview
            data = json.loads(content)
            text = " ".join([p["text"] for p in json.load(f)])


            themes = extract_key_themes(text)  # Reuse your theme extraction
            for theme in themes:
                theme_docs[theme].append(json_file.stem)
 
    # 2. Build network graph
    G = nx.Graph()
    for theme, docs in theme_docs.items():
        if len(docs) >= min_docs:  # Only significant themes
            G.add_node(theme, type="theme", size=len(docs)*5)
            for doc in docs:
                G.add_node(doc, type="document")
                G.add_edge(theme, doc)
    
    # 3. Convert to Plotly visualization
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        node_text.append(node)
        node_color.append("red" if G.nodes[node]["type"] == "theme" else "blue")
    
    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"),
            go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=node_text,
                marker=dict(color=node_color, size=20 if "theme" in node_text else 10),
                hoverinfo="text"
            )
        ],
        layout=go.Layout(showlegend=False, hovermode="closest")
    )
    return fig.to_json()
@app.get("/api/report/pdf")
async def generate_pdf_report():
    analyzer = ThemeAnalyzer(JSON_DIR)
    documents = analyzer.load_documents()
    themes = analyzer.analyze_themes_tfidf()
    reporter = ReportGenerator()
    pdf = await reporter.generate_and_return_pdf(themes, documents)
    return StreamingResponse(BytesIO(pdf), media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=theme_report.pdf"
    })