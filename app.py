from flask import Flask, request, render_template_string
import pysolr
import time

app = Flask(__name__)

# Connect to the local Solr instance you spun up via Docker
SOLR_URL = 'http://localhost:8983/solr/ev_reviews'
try:
    solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)
except Exception as e:
    print(f"Warning: Could not connect to Solr. Ensure Docker is running. Error: {e}")

# Embedded HTML Template (Using Bootstrap for immediate styling)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EV Opinion Search Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 20px; }
        .result-card { margin-bottom: 15px; border-left: 4px solid #0d6efd; }
        .highlight { background-color: #fff3cd; font-weight: bold; }
        .speed-metrics { font-size: 0.9em; color: #6c757d; margin-bottom: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1 class="mb-4 text-primary">🔋 EV Opinion Search</h1>
    
    <!-- Search Bar -->
    <form method="GET" action="/" class="mb-4">
        <div class="input-group input-group-lg">
            <input type="text" name="q" class="form-control" placeholder="Search EV opinions (e.g., battery degradation, autopilot)..." value="{{ current_query }}">
            <button class="btn btn-primary" type="submit">Search</button>
        </div>
        {% if current_platform %}
            <input type="hidden" name="platform" value="{{ current_platform }}">
        {% endif %}
    </form>

    <div class="row">
        <!-- Sidebar: Multifaceted Search (SC4021 Innovation) -->
        <div class="col-md-3">
            <div class="card mb-4">
                <div class="card-header bg-secondary text-white">Filter by Platform</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="/?q={{ current_query }}" class="text-decoration-none {% if not current_platform %}fw-bold text-dark{% endif %}">
                            All Platforms
                        </a>
                    </li>
                    {% for platform, count in facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="/?q={{ current_query }}&platform={{ platform }}" class="text-decoration-none {% if current_platform == platform %}fw-bold text-primary{% endif %}">
                                {{ platform }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>
            
            <!-- Future Placeholder for Sentiment Facets -->
            <div class="card mb-4">
                <div class="card-header bg-secondary text-white">Filter by Sentiment</div>
                <div class="card-body">
                    <p class="text-muted small"><em>Machine Learning labels will appear here after Phase 2 (Positive/Negative/Neutral).</em></p>
                </div>
            </div>
        </div>

        <!-- Main Results Area -->
        <div class="col-md-9">
            {% if num_found is not none %}
                <div class="speed-metrics">
                    Found <strong>{{ num_found }}</strong> results in <strong>{{ "%.3f"|format(query_time) }} seconds</strong>.
                </div>
                
                {% for doc in docs %}
                    <div class="card result-card shadow-sm">
                        <div class="card-body">
                            <h5 class="card-title">
                                {{ doc.source_target | default('General Topic', true) }} 
                                {% if doc.stars %}
                                    <span class="badge bg-warning text-dark float-end">⭐ {{ doc.stars }} / 5</span>
                                {% endif %}
                            </h5>
                            <h6 class="card-subtitle mb-2 text-muted">
                                {{ doc.platform | default('Unknown', true) }}
                                {% if doc.dataset_split == 'eval' %}
                                    <span class="badge bg-info text-dark ms-2">Eval Set</span>
                                {% endif %}
                            </h6>
                            <p class="card-text mt-3">"{{ doc.text | default('', true) }}"</p>
                        </div>
                    </div>
                {% else %}
                    <div class="alert alert-warning">No opinions found matching your query.</div>
                {% endfor %}
            {% else %}
                <div class="alert alert-info">Enter a query to start searching the database.</div>
            {% endif %}
        </div>
    </div>
</div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    # 1. Capture inputs
    raw_query = request.args.get('q', '').strip()
    platform_filter = request.args.get('platform', '')
    
    # Default to finding everything if search is empty
    solr_query = raw_query if raw_query else '*:*'
    
    # 2. Build Filter Queries (fq)
    fq =[]
    if platform_filter:
        # Wrap in quotes to handle platforms with spaces (e.g., "sgCarMart")
        fq.append(f'platform:"{platform_filter}"')
        
    # 3. Setup Faceting (The "Multifaceted Search" Innovation)
    params = {
        'facet': 'true',
        'facet.field': 'platform',
        'rows': 50, # Display up to 50 results per page
        'defType': 'edismax', # Robust parser that handles messy user input well
        'qf': 'text^2 source_target' # Boost hits found in the text field
    }
    
    docs =[]
    num_found = 0
    facets = {}
    query_time = 0.0
    
    try:
        # 4. Execute Search & Measure Speed (SC4021 Rubric Q2)
        start_time = time.time()
        results = solr.search(solr_query, fq=fq, **params)
        query_time = time.time() - start_time
        
        # 5. Parse Results
        num_found = results.hits
        docs = list(results)
        
        # Parse Solr facet arrays into a clean Python dictionary
        if 'facet_fields' in results.facets and 'platform' in results.facets['facet_fields']:
            raw_facets = results.facets['facet_fields']['platform']
            # Solr returns facets as a flat list:['Edmunds', 45, 'Trustpilot', 22]
            facets = {raw_facets[i]: raw_facets[i+1] for i in range(0, len(raw_facets), 2) if raw_facets[i+1] > 0}
            
    except Exception as e:
        print(f"Solr Search Error: {e}")
        num_found = None # Flags the UI to show an error or blank state

    # 6. Render the UI
    return render_template_string(
        HTML_TEMPLATE,
        docs=docs,
        num_found=num_found,
        query_time=query_time,
        current_query=raw_query,
        current_platform=platform_filter,
        facets=facets
    )

if __name__ == '__main__':
    # Run on port 5000. Debug mode auto-reloads the server when you make changes.
    app.run(host='0.0.0.0', port=8080, debug=True)