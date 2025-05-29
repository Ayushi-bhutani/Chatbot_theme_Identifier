from typing import List, Dict
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import os
from openai import AsyncOpenAI
import asyncio
from app.services.theme_analyzer import ThemeAnalyzer

# Load environment variables
load_dotenv()

class ReportGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_theme_report(self, themes: List[Dict], documents: List[Dict]) -> Dict:
        """Generate comprehensive theme analysis report"""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_documents": len(documents),
                "total_themes": len(themes)
            },
            "themes": await self._process_themes(themes),
            "statistics": self._calculate_stats(themes)
        }
        return report

    async def _process_themes(self, themes: List[Dict]) -> List[Dict]:
        """Generate AI summaries for each theme"""
        processed = []
        for theme in themes:
            summary = await self._generate_ai_summary(theme["name"], theme["documents"])
            processed.append({
                "theme": theme["name"],
                "document_count": len(theme["documents"]),
                "summary": summary,
                "representative_docs": theme["documents"][:3]
            })
        return processed

    async def _generate_ai_summary(self, theme: str, doc_ids: List[str]) -> str:
        """Generate GPT-4 summary of a theme"""
        prompt = f"""
        Analyze the theme '{theme}' across {len(doc_ids)} documents.
        Provide a 3–5 paragraph summary that:
        1. Defines the core concept
        2. Identifies key findings
        3. Notes any controversies
        4. Highlights significant evidence

        Write in an academic tone for medical researchers.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    def _calculate_stats(self, themes: List[Dict]) -> Dict:
        """Calculate theme statistics"""
        theme_counts = [len(t["documents"]) for t in themes]
        unique_docs = len(set(d for t in themes for d in t["documents"]))
        return {
            "most_common_theme": max(themes, key=lambda x: len(x["documents"]))["name"],
            "avg_docs_per_theme": sum(theme_counts) / len(theme_counts) if theme_counts else 0,
            "theme_coverage": sum(theme_counts) / unique_docs if unique_docs else 0
        }

    async def generate_pdf(self, report: Dict) -> bytes:
        """Generate PDF using ReportLab"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph(
            f"Theme Analysis Report - {report['metadata']['generated_at']}",
            styles['Title']
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Add statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        stats = [
            f"Most common theme: {report['statistics']['most_common_theme']}",
            f"Average documents per theme: {report['statistics']['avg_docs_per_theme']:.1f}",
            f"Total themes: {report['metadata']['total_themes']}",
            f"Total documents: {report['metadata']['total_documents']}"
        ]
        for stat in stats:
            story.append(Paragraph(stat, styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        # Add themes
        story.append(Paragraph("Theme Details", styles['Heading2']))
        for theme in report['themes']:
            story.append(Paragraph(
                f"{theme['theme']} ({theme['document_count']} documents)",
                styles['Heading3']
            ))
            story.append(Paragraph(theme['summary'], styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    async def generate_and_return_pdf(self, themes: List[Dict], documents: List[Dict]) -> bytes:
        """Full wrapper: Generate report then PDF"""
        report = await self.generate_theme_report(themes, documents)
        pdf = await self.generate_pdf(report)
        return pdf
    theme_analyzer = ThemeAnalyzer(json_dir="../data/extracted_json")
