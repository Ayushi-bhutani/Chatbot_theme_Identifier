import os
from dotenv import load_dotenv

# ✅ Load .env variables first
load_dotenv()
from app.services.vector_db import VectorDB
from app.services.gpt_summarizer import generate_theme_summary
vector_db = VectorDB() 
import openai
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List
from collections import defaultdict
import shutil
import pytesseract
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF
import json
import re
import traceback
# ✅ Load environment variables early
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
print("Loaded OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

# ✅ Now use the loaded variables
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI Key:", openai.api_key)  # Optional for debug

# ✅ Handle missing key
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set!")

_themes_cache = []

app = FastAPI()

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict origins in production!
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

# --- Pydantic Models for theme summary endpoint ---

class ThemeSummaryRequest(BaseModel):
    theme: str
    documents: List[str]
    summary_snippets: List[str]


class ThemeResponse(BaseModel):
    theme: str
    gpt_summary: str
# Add these imports at the top if not already present

import magic  # python-magic package
from typing import Optional
from datetime import datetime
import nltk
nltk.download('punkt')
from typing import List
from pathlib import Path
from datetime import datetime
import pdfplumber
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def extract_text_from_pdf(pdf_path: Path) -> List[dict]:
    """Enhanced PDF text extraction with metadata, sentence tokenization, and better OCR handling"""
    text_data = []
    metadata = {
        "filename": pdf_path.name,
        "upload_date": datetime.now().isoformat(),
        "page_count": 0,
        "is_scanned": False
    }
    
    try:
        # First try with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            metadata["page_count"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                content = page.extract_text()
                if not content:
                    # If no text found, try harder with alternative extraction
                    content = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if not content:
                        metadata["is_scanned"] = True
                        raise ValueError("Possible scanned document")
                
                # Split page text into sentences
                sentences = sent_tokenize(content.strip())
                for sent in sentences:
                    text_data.append({
                        "page": i,
                        "text": sent,
                        "sentence_id": len(text_data) + 1,
                        "bbox": page.bbox
                    })
                
    except Exception as e:
        print(f"[INFO] Falling back to OCR processing: {str(e)}")
        text_data = ocr_scanned_pdf(pdf_path)
        metadata["is_scanned"] = True
    
    # Add metadata to the first sentence/page entry
    if text_data:
        text_data[0]["metadata"] = metadata
    
    return text_data

def ocr_scanned_pdf(pdf_path: Path) -> List[dict]:
    """Improved OCR processing with better image handling"""
    text_by_page = []
    doc = fitz.open(pdf_path)
    
    for i, page in enumerate(doc, start=1):
        try:
            # Get page as image with higher DPI for better OCR
            pix = page.get_pixmap(dpi=300)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Preprocess image for better OCR results
            image = image.convert('L')  # Grayscale
            image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Thresholding
            
            # Custom OCR configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Split OCR text into sentences as well
            sentences = sent_tokenize(text.strip())
            for sent in sentences:
                text_by_page.append({
                    "page": i,
                    "text": sent,
                    "ocr_processed": True
                })
        except Exception as e:
            print(f"Error processing page {i}: {str(e)}")
            text_by_page.append({
                "page": i,
                "text": "",
                "error": str(e)
            })
    
    return text_by_page


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type using magic numbers
    file_content = await file.read(2048)
    await file.seek(0)
    
    mime_type = magic.from_buffer(file_content, mime=True)
    if mime_type != 'application/pdf':
        raise HTTPException(400, "Only PDF files are accepted")
    
    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(400, "File too large. Max 50MB allowed")
    
    pdf_path = PDF_DIR / file.filename
    
    # Check for existing file
    if pdf_path.exists():
        raise HTTPException(400, "File with this name already exists")
    
    # Save the file
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Validate PDF structure
    try:
        with fitz.open(pdf_path) as doc:
            if not doc.is_pdf:
                os.remove(pdf_path)
                raise HTTPException(400, "Invalid PDF structure")
    except:
        os.remove(pdf_path)
        raise HTTPException(400, "Invalid PDF file")
    
    # Process the PDF
    try:
        extracted_data = extract_text_from_pdf(pdf_path)
        
        # Save extracted text as JSON
        json_path = JSON_DIR / f"{file.filename}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(extracted_data, jf, ensure_ascii=False, indent=2)
        
        # Index in vector database (new addition)
        try:
            vector_db.index_document(file.filename, extracted_data)
        except Exception as vec_err:
            print(f"Vector DB indexing error: {vec_err}")
            # Don't fail the upload, just log the error
            # You might want to implement retry logic here
        
        return {
            "filename": file.filename,
            "message": "Uploaded, text extracted, and indexed successfully.",
            "metadata": extracted_data[0].get("metadata", {}) if extracted_data else {},
            "vector_status": "indexed"  # new field to indicate vector DB status
        }
    except Exception as e:
        # Cleanup in case of failure
        if pdf_path.exists():
            os.remove(pdf_path)
        json_path = JSON_DIR / f"{file.filename}.json"
        if json_path.exists():
            os.remove(json_path)
        
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")


# --- Query documents endpoint ---
@app.post("/query/")
def query_documents(
    question: str = Query(..., min_length=3, description="Your search query"),
    semantic: bool = Query(True, description="Use semantic search (vector DB)"),
    keyword: bool = Query(False, description="Also include keyword matches"),
    limit: int = Query(5, description="Number of results to return")
):
    """
    Search across documents using either semantic search (default) or keyword matching.
    Returns results with citations and relevance scores.
    """
    results = []
    
    # Semantic Search (Vector DB)
    if semantic:
        try:
            vector_results = vector_db.search(question, limit=limit)
            for res in vector_results:
                results.append({
                    "document": res["doc_id"],
                    "page": res["page"],
                    "excerpt": highlight_query(res["text"], question),
                    "score": float(res["score"]),
                    "type": "semantic",
                    "citation": f"{res['doc_id']}, Page {res['page']}"
                })
        except Exception as e:
            print(f"Vector search error: {e}")
            if not keyword:  # Only fail if keyword search also disabled
                raise HTTPException(500, "Semantic search temporarily unavailable")

    # Keyword Search (Fallback or hybrid mode)
    if keyword:
        for file in JSON_DIR.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                pages = json.load(f)
                for page in pages:
                    text = page.get("text", "").lower()
                    if question.lower() in text:
                        results.append({
                            "document": file.stem.replace(".pdf", ""),
                            "page": page["page"],
                            "excerpt": highlight_query(page["text"], question),
                            "score": 0.5,  # Default score for keyword matches
                            "type": "keyword",
                            "citation": f"{file.stem.replace('.pdf', '')}, Page {page['page']}"
                        })

    # Sort combined results by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply limit
    results = results[:limit]
    
    return {
        "query": question,
        "results": results,
        "count": len(results)
    }

def highlight_query(text: str, query: str, snippet_length: int = 300) -> str:
    """
    Returns a text snippet with query terms highlighted (case insensitive)
    """
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Find first occurrence
    start_pos = text_lower.find(query_lower)
    if start_pos == -1:
        return text[:snippet_length] + "..." if len(text) > snippet_length else text
    
    # Get surrounding context
    snippet_start = max(0, start_pos - snippet_length//2)
    snippet_end = min(len(text), start_pos + len(query) + snippet_length//2)
    snippet = text[snippet_start:snippet_end]
    
    # Highlight the query terms
    for term in query.split():
        if len(term) > 3:  # Only highlight significant terms
            snippet = re.sub(
                f"({term})", 
                r"[**\1**]", 
                snippet, 
                flags=re.IGNORECASE
            )
    
    return ("..." if snippet_start > 0 else "") + snippet + ("..." if snippet_end < len(text) else "")
# --- Synthesize themes endpoint ---
@app.get("/synthesize/")
def synthesize_themes():
    word_doc_map = defaultdict(set)
    doc_summaries = {}

    for json_file in JSON_DIR.glob("*.json"):
        doc_id = json_file.stem.replace(".pdf", "")
        with open(json_file, "r", encoding="utf-8") as f:
            pages = json.load(f)

        all_text = " ".join([p["text"] for p in pages if p.get("text")])
        doc_summaries[doc_id] = all_text[:500] + "..."
        words = re.findall(r'\b\w{6,}\b', all_text.lower())  # Words of length >=6

        for word in words:
            word_doc_map[word].add(doc_id)

    # Filter words appearing in multiple documents as themes
    theme_candidates = {w: list(docs) for w, docs in word_doc_map.items() if len(docs) > 1}

    # Build themes list
    themes = []
    for theme_word, docs in list(theme_candidates.items())[:5]:  # limit to 5 themes
        summary_snippets = []
        for doc_id in docs:
            summary_snippets.append(f"{doc_id}: {doc_summaries[doc_id]}")
        themes.append({
            "theme": theme_word,
            "documents": docs,
            "summary_snippets": summary_snippets
        })

    # Save themes globally for use in /themes endpoint
    global _themes_cache
    _themes_cache = themes

    return {"themes_found": len(themes), "themes": themes}

# --- Get themes with GPT summaries ---

# def generate_theme_summary(theme: str, snippets: List[str]) -> str:
#     prompt = (
#         f"Summarize the following scientific literature snippets related to the theme '{theme}' "
#         f"into a concise, informative paragraph:\n\n"
#         + "\n\n".join(snippets)
#     )
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert scientific summarizer."},
#                 {"role": "user", "content": prompt},
#             ],
#             max_tokens=300,
#             temperature=0.5,
#         )
#         summary = response.choices[0].message.content.strip()
#         return summary
#     except Exception as e:
#         print("OpenAI API error:", e)
#         traceback.print_exc()
#         return "Summary could not be generated at this time."


# @app.post("/summarize_theme/", response_model=ThemeResponse)
# async def summarize_theme(request: ThemeSummaryRequest):
#     if not request.summary_snippets or not request.theme:
#         raise HTTPException(status_code=422, detail="Theme, documents and summary_snippets must be provided and non-empty.")
#     # Now you can access request.documents if needed
#     summary = generate_theme_summary(request.theme, request.summary_snippets)
#     return ThemeResponse(theme=request.theme, gpt_summary=summary)


@app.post("/summarize_theme/", response_model=ThemeResponse)
async def summarize_theme(request: ThemeSummaryRequest):
    if not request.summary_snippets or not request.theme:
        raise HTTPException(status_code=422, detail="Theme, documents and summary_snippets must be provided and non-empty.")
    
    summary = generate_theme_summary(request.theme, request.summary_snippets)
    return ThemeResponse(theme=request.theme, gpt_summary=summary)

# Add this temporary debug endpoint to backend/app/main.py
@app.get("/debug/{filename}")
def debug_document(filename: str):
    json_path = JSON_DIR / f"{filename}.json"
    if not json_path.exists():
        return {"error": "File not processed"}
    
    with open(json_path, "r") as f:
        content = json.load(f)
    
    # Check if vector DB has this doc
    vector_results = vector_db.search("liver", limit=1)
    return {
        "text_extracted": bool(content),
        "vector_db_has_doc": any(r["doc_id"] == filename for r in vector_results),
        "first_page_text": content[0]["text"][:200] if content else None
    }
# Temporary reindex endpoint
@app.post("/reindex/")
def reindex_all():
    for pdf in PDF_DIR.glob("*.pdf"):
        extracted = extract_text_from_pdf(pdf)
        vector_db.index_document(pdf.name, extracted)
    return {"status": "reindexed"}