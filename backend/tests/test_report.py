import asyncio
from pathlib import Path
from app.services.theme_analyzer import ThemeAnalyzer
from app.services.report_generator import ReportGenerator

async def main():
    analyzer = ThemeAnalyzer(Path("data/extracted_json"))
    documents = [...]  # Your document loading logic here
    themes = analyzer.extract_common_themes(documents)
    
    reporter = ReportGenerator()
    
    # Test JSON report
    report = await reporter.generate_theme_report(themes, documents)
    print("Report generated successfully!")
    print(f"Found {len(report['themes'])} themes")
    
    # Test PDF generation
    pdf_bytes = await reporter.generate_pdf(report)
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF saved to test_report.pdf")
    

if __name__ == "__main__":
    asyncio.run(main())