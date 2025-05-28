# Standard Library
import os
import re
import sys
import json
import time
import uuid
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime
from hashlib import sha256

# Third Party
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pytesseract
import pdfplumber
from PIL import Image
import fitz
import magic
from faker import Faker
from openai import AsyncOpenAI

# Local
from app.schemas import QueryRequest
# from app.services.vector_db import VectorDB
from app.services.theme_analyzer import ThemeAnalyzer