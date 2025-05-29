from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import json
from typing import List, Dict, DefaultDict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
router = APIRouter()
def generate_d3_visualization(themes: List[Dict], documents: List[Dict]) -> str:
    """Generate HTML with D3.js visualization"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Theme Network</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            #network {{ 
                width: 100%; 
                height: 80vh;
                border: 1px solid #eee;
                margin-top: 20px;
            }}
            .node.theme {{ fill: #ff7f0e; stroke: #cc6600; }}
            .node.document {{ fill: #1f77b4; stroke: #0d5ba3; }}
            .link {{ stroke: #999; stroke-opacity: 0.6; }}
            .tooltip {{
                position: absolute;
                padding: 8px;
                background: rgba(0,0,0,0.8);
                color: white;
                border-radius: 4px;
                pointer-events: none;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <h1>Document Theme Network</h1>
        <div id="network"></div>
        <div class="tooltip" id="tooltip"></div>
        

        <script>
            // 1. Prepare the data
            const themes = {json.dumps(themes)};
            const documents = {json.dumps(documents)};
            
            // 2. Create the force-directed graph
            const width = document.getElementById('network').clientWidth;
            const height = 600;
            
            const svg = d3.select("#network")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // Create simulation
            const simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));
            
            // 3. Process nodes and links
            const themeNodes = themes.map(t => ({{
                ...t,
                type: "theme",
                radius: 10 + Math.sqrt(t.count) * 2
            }}));
            
            const docNodes = documents.map(d => ({{
                ...d,
                type: "document",
                radius: 8
            }}));
            
            const allNodes = [...themeNodes, ...docNodes];
            
            const links = [];
            documents.forEach(doc => {{
                doc.themes.forEach(themeId => {{
                    links.push({{
                        source: doc.id,
                        target: themeId,
                        value: 1
                    }});
                }});
            }});
            
            // 4. Draw the graph
            const link = svg.append("g")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.value)));
            
            const node = svg.append("g")
                .selectAll("circle")
                .data(allNodes)
                .enter().append("circle")
                .attr("class", d => `node ${{d.type}}`)
                .attr("r", d => d.radius)
                .call(drag(simulation)));
            
            // 5. Add interactivity
            const tooltip = d3.select("#tooltip");
            
            node.on("mouseover", (event, d) => {{
                tooltip
                    .style("opacity", 1)
                    .html(`<strong>${{d.name || d.title}}</strong><br>
                          ${{d.type === 'theme' ? `Documents: ${{d.count}}` : `Themes: ${{d.themes.length}}`}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", () => {{
                tooltip.style("opacity", 0);
            }});
            
            // 6. Update simulation
            simulation
                .nodes(allNodes)
                .on("tick", ticked);
            
            simulation.force("link")
                .links(links);
            
            function ticked() {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            }}
            
            function drag(simulation) {{
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
                
                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
                
            }}
        </script>
    </body>
    </html>
    """
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
@router.get("/visualize", response_class=HTMLResponse)
async def visualize_themes(search: str = None, show_themes: bool = True, show_docs: bool = True):
    """Endpoint to generate theme visualization"""
    # 1. Extract themes from all documents
    theme_docs = defaultdict(list)
    documents = []
    json_dir = Path("data/extracted_json")
    
    for json_file in json_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            text = " ".join([p["text"] for p in data["pages"]])
            doc_id = json_file.stem
            themes = extract_key_themes(text)  # Using our new function
            
            documents.append({
                "id": doc_id,
                "title": doc_id.replace("_", " "),
                "themes": themes
            })
            
            for theme in themes:
                theme_docs[theme].append(doc_id)
    
    # 2. Prepare visualization data
    viz_themes = [
        {
            "id": f"theme_{i}",
            "name": theme,
            "count": len(docs),
            "documents": docs
        }
        for i, (theme, docs) in enumerate(theme_docs.items())
        if len(docs) >= 2  # Only show themes appearing in multiple docs
    ]
    
    viz_documents = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "themes": [t["id"] for t in viz_themes if t["name"] in doc["themes"]]
        }
        for doc in documents
    ]
    
    # 3. Generate and return the visualization
    html = generate_d3_visualization(viz_themes, viz_documents)
    
    # Add URL parameter support
    html = html.replace(
        '<script>',
        f'<script>\nconst urlParams = {{\n'
        f'  search: {"null" if not search else f'"{search}"'},\n'
        f'  showThemes: {"true" if show_themes else "false"},\n'
        f'  showDocs: {"true" if show_docs else "false"}\n}};'
    )
    
    return HTMLResponse(content=html)
def generate_filter_controls() -> str:
    """Generate HTML filter controls"""
    return """
    <div class="controls">
        <input type="text" id="search" placeholder="Search themes/documents...">
        <div class="filters">
            <label>
                <input type="checkbox" class="filter" value="theme" checked> Themes
            </label>
            <label>
                <input type="checkbox" class="filter" value="document" checked> Documents
            </label>
        </div>
    </div>
    <style>
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        #search {
            padding: 8px;
            width: 300px;
        }
        .filters {
            display: flex;
            gap: 15px;
        }
    </style>
    """