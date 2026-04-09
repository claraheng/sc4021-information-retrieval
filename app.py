from flask import Flask, request, render_template_string
import pysolr
import time
import re
from markupsafe import Markup, escape
from urllib.parse import urlencode

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
        .pagination-wrap { margin-top: 25px; }
        .facet-card .list-group-item { font-size: 0.95rem; }
        .summary-card { height: 100%; }
        .summary-bar { height: 10px; }
        .chip-link { margin: 0 8px 8px 0; }
        .badge-positive { background-color: #198754; }
        .badge-negative { background-color: #dc3545; }
        .badge-neutral { background-color: #6c757d; }
        .badge-opinionated { background-color: #0d6efd; }
        .badge-subj-neutral { background-color: #adb5bd; color: #212529; }
        .interactive-card { border: 1px solid #dfe6ee; }
        .interactive-section-title { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #6c757d; }
        .interactive-pill { display: inline-flex; align-items: center; border: 1px solid #ced4da; border-radius: 999px; padding: 8px 12px; margin: 0 10px 10px 0; text-decoration: none; color: #212529; background: #fff; transition: all 0.15s ease; }
        .interactive-pill:hover { background: #f1f5f9; border-color: #0d6efd; color: #0d6efd; }
        .interactive-pill strong { margin-right: 6px; }
        .interactive-note { font-size: 0.92rem; color: #6c757d; }
        .filter-chip { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; background: #212529; color: #fff; padding: 6px 10px; margin: 0 8px 8px 0; font-size: 0.9rem; }
        .filter-chip a { color: #fff; text-decoration: none; font-weight: 700; }
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
        {% if current_source_target %}
            <input type="hidden" name="source_target" value="{{ current_source_target }}">
        {% endif %}
        {% if current_dataset_split %}
            <input type="hidden" name="dataset_split" value="{{ current_dataset_split }}">
        {% endif %}
        {% if current_polarity %}
            <input type="hidden" name="polarity" value="{{ current_polarity }}">
        {% endif %}
        {% if current_subjectivity %}
            <input type="hidden" name="subjectivity" value="{{ current_subjectivity }}">
        {% endif %}
    </form>

    <div class="row">
        <!-- Sidebar: Multifaceted Search (SC4021 Innovation) -->
        <div class="col-md-3">
            <div class="card mb-4 facet-card">
                <div class="card-header bg-secondary text-white">Filter by Platform</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="{{ build_url(platform='') }}" class="text-decoration-none {% if not current_platform %}fw-bold text-dark{% endif %}">
                            All Platforms
                        </a>
                    </li>
                    {% for platform, count in facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="{{ build_url(platform=platform) }}" class="text-decoration-none {% if current_platform == platform %}fw-bold text-primary{% endif %}">
                                {{ platform }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="card mb-4 facet-card">
                <div class="card-header bg-secondary text-white">Filter by Source/Target</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="{{ build_url(source_target='') }}" class="text-decoration-none {% if not current_source_target %}fw-bold text-dark{% endif %}">
                            All Targets
                        </a>
                    </li>
                    {% for source_target, count in source_target_facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="{{ build_url(source_target=source_target) }}" class="text-decoration-none {% if current_source_target == source_target %}fw-bold text-primary{% endif %}">
                                {{ source_target }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>

            <div class="card mb-4 facet-card">
                <div class="card-header bg-secondary text-white">Filter by Dataset Split</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="{{ build_url(dataset_split='') }}" class="text-decoration-none {% if not current_dataset_split %}fw-bold text-dark{% endif %}">
                            All Documents
                        </a>
                    </li>
                    {% for split_name, count in dataset_split_facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="{{ build_url(dataset_split=split_name) }}" class="text-decoration-none {% if current_dataset_split == split_name %}fw-bold text-primary{% endif %}">
                                {{ split_name }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>

            <div class="card mb-4 facet-card">
                <div class="card-header bg-secondary text-white">Filter by Polarity</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="{{ build_url(polarity='') }}" class="text-decoration-none {% if not current_polarity %}fw-bold text-dark{% endif %}">
                            All Sentiments
                        </a>
                    </li>
                    {% for polarity_name, count in polarity_facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="{{ build_url(polarity=polarity_name) }}" class="text-decoration-none {% if current_polarity == polarity_name %}fw-bold text-primary{% endif %}">
                                {{ polarity_name }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>

            <div class="card mb-4 facet-card">
                <div class="card-header bg-secondary text-white">Filter by Subjectivity</div>
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">
                        <a href="{{ build_url(subjectivity='') }}" class="text-decoration-none {% if not current_subjectivity %}fw-bold text-dark{% endif %}">
                            All Voice Types
                        </a>
                    </li>
                    {% for subjectivity_name, count in subjectivity_facets.items() %}
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="{{ build_url(subjectivity=subjectivity_name) }}" class="text-decoration-none {% if current_subjectivity == subjectivity_name %}fw-bold text-primary{% endif %}">
                                {{ subjectivity_name }}
                            </a>
                            <span class="badge bg-primary rounded-pill">{{ count }}</span>
                        </li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <!-- Main Results Area -->
        <div class="col-md-9">
            {% if num_found is not none %}
                <div class="speed-metrics">
                    Found <strong>{{ num_found }}</strong> results in <strong>{{ "%.3f"|format(query_time) }} seconds</strong>.
                    Showing page <strong>{{ current_page }}</strong> of <strong>{{ total_pages }}</strong>.
                </div>

                {% if active_filters %}
                    <div class="mb-3">
                        {% for filter_label in active_filters %}
                            <span class="filter-chip">{{ filter_label.label }} <a href="{{ filter_label.clear_url }}">×</a></span>
                        {% endfor %}
                        <a href="{{ build_url(platform='', source_target='', dataset_split='', polarity='', subjectivity='') }}" class="btn btn-sm btn-outline-secondary">Clear Filters</a>
                    </div>
                {% endif %}

                <div class="row g-3 mb-4">
                    <div class="col-md-12">
                        <div class="card summary-card shadow-sm">
                            <div class="card-header bg-light">Main Goal: Sentiment of Current Results</div>
                            <div class="card-body">
                                {% if sentiment_summary %}
                                    <div class="row g-3">
                                        {% for item in sentiment_summary %}
                                            <div class="col-md-4">
                                                <div class="border rounded p-3 h-100">
                                                    <div class="d-flex justify-content-between align-items-center">
                                                        <span class="fw-semibold">{{ item.label }}</span>
                                                        <span class="badge {{ item.badge_class }}">{{ item.count }}</span>
                                                    </div>
                                                    <div class="small text-muted mt-2">{{ item.percent }}% of current result set</div>
                                                    <div class="progress summary-bar mt-2">
                                                        <div class="progress-bar {{ item.progress_class }}" role="progressbar" style="width: {{ item.percent }}%"></div>
                                                    </div>
                                                </div>
                                            </div>
                                        {% endfor %}
                                    </div>
                                    {% if dominant_sentiment %}
                                        <p class="mb-0 mt-3">
                                            Overall sentiment trend for this query:
                                            <strong>{{ dominant_sentiment.label }}</strong>
                                            ({{ dominant_sentiment.percent }}% of current matches).
                                        </p>
                                    {% endif %}
                                {% else %}
                                    <p class="text-muted mb-0">No sentiment summary available for this query.</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card summary-card shadow-sm">
                            <div class="card-header bg-light">Enhanced Search: Platform Mix</div>
                            <div class="card-body">
                                {% if platform_summary %}
                                    {% for item in platform_summary %}
                                        <div class="d-flex justify-content-between small">
                                            <span>{{ item.label }}</span>
                                            <span>{{ item.count }} ({{ item.percent }}%)</span>
                                        </div>
                                        <div class="progress summary-bar mb-3">
                                            <div class="progress-bar" role="progressbar" style="width: {{ item.percent }}%"></div>
                                        </div>
                                    {% endfor %}
                                {% else %}
                                    <p class="text-muted mb-0">No platform summary available.</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card summary-card shadow-sm">
                            <div class="card-header bg-light">Enhanced Search: Target Mix</div>
                            <div class="card-body">
                                {% if source_target_summary %}
                                    {% for item in source_target_summary %}
                                        <div class="d-flex justify-content-between small">
                                            <span>{{ item.label }}</span>
                                            <span>{{ item.count }} ({{ item.percent }}%)</span>
                                        </div>
                                        <div class="progress summary-bar mb-3">
                                            <div class="progress-bar bg-success" role="progressbar" style="width: {{ item.percent }}%"></div>
                                        </div>
                                    {% endfor %}
                                {% else %}
                                    <p class="text-muted mb-0">No target summary available.</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>

                {% if quick_refine_links %}
                    <div class="card mb-4 shadow-sm interactive-card">
                        <div class="card-header bg-light">Interactive Search: Guided Refinement</div>
                        <div class="card-body">
                            <p class="interactive-note mb-3">Narrow the current results without rewriting your query. These suggestions are based on the strongest patterns in the current result set.</p>
                            {% for group in quick_refine_links %}
                                <div class="mb-3">
                                    <div class="interactive-section-title mb-2">{{ group.title }}</div>
                                    {% for item in group.options %}
                                        <a class="interactive-pill" href="{{ item.url }}">
                                            <strong>{{ item.label }}</strong>
                                            <span class="text-muted">{{ item.meta }}</span>
                                        </a>
                                    {% endfor %}
                                </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}
                
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
                                {% if doc.subjectivity %}
                                    <span class="badge {{ doc.subjectivity_badge_class }} ms-2">{{ doc.subjectivity }}</span>
                                {% endif %}
                                {% if doc.polarity %}
                                    <span class="badge {{ doc.polarity_badge_class }} ms-2">{{ doc.polarity }}</span>
                                {% endif %}
                            </h6>
                            <p class="card-text mt-3">"{{ doc.highlighted_text | default(doc.text, true) }}"</p>
                        </div>
                    </div>
                {% else %}
                    <div class="alert alert-warning">No opinions found matching your query.</div>
                {% endfor %}

                {% if total_pages > 1 %}
                    <nav class="pagination-wrap" aria-label="Search results pages">
                        <ul class="pagination flex-wrap">
                            <li class="page-item {% if current_page <= 1 %}disabled{% endif %}">
                                <a class="page-link" href="{{ build_url(page=current_page - 1) }}">Previous</a>
                            </li>
                            {% for page_num in page_numbers %}
                                <li class="page-item {% if page_num == current_page %}active{% endif %}">
                                    <a class="page-link" href="{{ build_url(page=page_num) }}">{{ page_num }}</a>
                                </li>
                            {% endfor %}
                            <li class="page-item {% if current_page >= total_pages %}disabled{% endif %}">
                                <a class="page-link" href="{{ build_url(page=current_page + 1) }}">Next</a>
                            </li>
                        </ul>
                    </nav>
                {% endif %}
            {% else %}
                <div class="alert alert-info">Enter a query to start searching the database.</div>
            {% endif %}
        </div>
    </div>
</div>
</body>
</html>
"""


def build_highlight_pattern(raw_query):
    terms = []
    for token in re.findall(r'"([^"]+)"|(\S+)', raw_query):
        value = (token[0] or token[1]).strip()
        value = re.sub(r"[+\-()]", " ", value)
        value = value.strip()
        if len(value) < 2:
            continue
        terms.extend(part for part in value.split() if len(part) >= 2)

    deduped_terms = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped_terms.append(re.escape(term))

    if not deduped_terms:
        return None

    return re.compile(r"(" + "|".join(deduped_terms) + r")", re.IGNORECASE)


def normalize_text_value(text):
    if text is None:
        return ""
    if isinstance(text, list):
        return " ".join(str(part) for part in text if part is not None)
    return str(text)


def highlight_text(text, raw_query):
    text = normalize_text_value(text)

    if not text or not raw_query:
        return escape(text or "")

    pattern = build_highlight_pattern(raw_query)
    if pattern is None:
        return escape(text)

    parts = []
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        if start > last_end:
            parts.append(escape(text[last_end:start]))
        parts.append(Markup('<span class="highlight">'))
        parts.append(escape(match.group(0)))
        parts.append(Markup("</span>"))
        last_end = end

    if last_end == 0:
        return escape(text)

    if last_end < len(text):
        parts.append(escape(text[last_end:]))

    return Markup("").join(parts)


def get_page_window(current_page, total_pages, window_size=7):
    if total_pages <= window_size:
        return list(range(1, total_pages + 1))

    half_window = window_size // 2
    start_page = max(1, current_page - half_window)
    end_page = min(total_pages, start_page + window_size - 1)

    if end_page - start_page + 1 < window_size:
        start_page = max(1, end_page - window_size + 1)

    return list(range(start_page, end_page + 1))


def build_search_url(raw_query="", platform="", source_target="", dataset_split="", polarity="", subjectivity="", page=1):
    params = {}
    if raw_query:
        params["q"] = raw_query
    if platform:
        params["platform"] = platform
    if source_target:
        params["source_target"] = source_target
    if dataset_split:
        params["dataset_split"] = dataset_split
    if polarity:
        params["polarity"] = polarity
    if subjectivity:
        params["subjectivity"] = subjectivity
    if page and page != 1:
        params["page"] = page
    query_string = urlencode(params)
    return f"/?{query_string}" if query_string else "/"


def parse_facet_counts(results, field_name):
    facet_fields = getattr(results, "facets", {}).get("facet_fields", {})
    raw_values = facet_fields.get(field_name, [])
    return {
        raw_values[i]: raw_values[i + 1]
        for i in range(0, len(raw_values), 2)
        if raw_values[i + 1] > 0
    }


def build_summary_items(facet_counts, total_hits, limit=5):
    if total_hits <= 0:
        return []

    items = []
    for label, count in list(facet_counts.items())[:limit]:
        percent = round((count / total_hits) * 100, 1)
        items.append({
            "label": label,
            "count": count,
            "percent": percent,
        })
    return items


def sentiment_badge_class(label):
    label = (label or "").strip().lower()
    if label == "positive":
        return "badge-positive"
    if label == "negative":
        return "badge-negative"
    if label == "neutral":
        return "badge-neutral"
    if label == "opinionated":
        return "badge-opinionated"
    return "badge-subj-neutral"


def build_sentiment_summary(polarity_facets, total_hits):
    items = []
    progress_classes = {
        "positive": "bg-success",
        "negative": "bg-danger",
        "neutral": "bg-secondary",
    }
    for label in ["positive", "negative", "neutral"]:
        count = polarity_facets.get(label, 0)
        percent = round((count / total_hits) * 100, 1) if total_hits else 0
        items.append({
            "label": label.title(),
            "count": count,
            "percent": percent,
            "badge_class": sentiment_badge_class(label),
            "progress_class": progress_classes[label],
        })
    return items


def dominant_sentiment_from_facets(polarity_facets, total_hits):
    if total_hits <= 0 or not polarity_facets:
        return None
    label, count = max(polarity_facets.items(), key=lambda item: item[1])
    return {
        "label": label.title(),
        "count": count,
        "percent": round((count / total_hits) * 100, 1),
    }


def build_quick_refine_links(raw_query, platform_facets, source_target_facets, polarity_facets, current_platform, current_source_target, current_dataset_split, current_polarity, current_subjectivity):
    groups = []
    platform_items = []
    target_items = []
    polarity_items = []

    if not current_platform:
        for platform in list(platform_facets.keys())[:3]:
            platform_items.append({
                "label": f"Only {platform}",
                "meta": f"{platform_facets[platform]} matching results",
                "url": build_search_url(
                    raw_query=raw_query,
                    platform=platform,
                    source_target=current_source_target,
                    dataset_split=current_dataset_split,
                    polarity=current_polarity,
                    subjectivity=current_subjectivity,
                ),
            })

    if not current_source_target:
        for source_target in list(source_target_facets.keys())[:3]:
            target_items.append({
                "label": f"{source_target}",
                "meta": f"{source_target_facets[source_target]} matching results",
                "url": build_search_url(
                    raw_query=raw_query,
                    platform=current_platform,
                    source_target=source_target,
                    dataset_split=current_dataset_split,
                    polarity=current_polarity,
                    subjectivity=current_subjectivity,
                ),
            })

    if not current_polarity:
        for polarity in list(polarity_facets.keys())[:3]:
            polarity_items.append({
                "label": polarity.title(),
                "meta": f"{polarity_facets[polarity]} matching results",
                "url": build_search_url(
                    raw_query=raw_query,
                    platform=current_platform,
                    source_target=current_source_target,
                    dataset_split=current_dataset_split,
                    polarity=polarity,
                    subjectivity=current_subjectivity,
                ),
            })

    if platform_items:
        groups.append({
            "title": "Refine by Platform",
            "options": platform_items,
        })
    if target_items:
        groups.append({
            "title": "Refine by Source/Target",
            "options": target_items,
        })
    if polarity_items:
        groups.append({
            "title": "Refine by Sentiment",
            "options": polarity_items,
        })

    return groups[:3]

@app.route('/', methods=['GET'])
def index():
    # 1. Capture inputs
    raw_query = request.args.get('q', '').strip()
    platform_filter = request.args.get('platform', '')
    source_target_filter = request.args.get('source_target', '').strip()
    dataset_split_filter = request.args.get('dataset_split', '').strip()
    polarity_filter = request.args.get('polarity', '').strip().lower()
    subjectivity_filter = request.args.get('subjectivity', '').strip().lower()
    try:
        current_page = max(1, int(request.args.get('page', '1')))
    except ValueError:
        current_page = 1
    
    # Default to finding everything if search is empty
    solr_query = raw_query if raw_query else '*:*'
    rows_per_page = 10
    start_offset = (current_page - 1) * rows_per_page
    
    # 2. Build Filter Queries (fq)
    fq =[]
    if platform_filter:
        # Wrap in quotes to handle platforms with spaces (e.g., "sgCarMart")
        fq.append(f'platform:"{platform_filter}"')
    if source_target_filter:
        fq.append(f'source_target:"{source_target_filter}"')
    if dataset_split_filter:
        fq.append(f'dataset_split:"{dataset_split_filter}"')
    if polarity_filter:
        fq.append(f'polarity:"{polarity_filter}"')
    if subjectivity_filter:
        fq.append(f'subjectivity:"{subjectivity_filter}"')
        
    # 3. Setup Faceting (The "Multifaceted Search" Innovation)
    params = {
        'facet': 'true',
        'facet.field': ['platform', 'source_target', 'dataset_split', 'polarity', 'subjectivity'],
        'facet.mincount': 1,
        'facet.limit': 8,
        'rows': rows_per_page,
        'start': start_offset,
        'defType': 'edismax', # Robust parser that handles messy user input well
        'qf': 'text^2 source_target', # Boost hits found in the text field
    }
    
    docs =[]
    num_found = 0
    facets = {}
    query_time = 0.0
    total_pages = 1
    page_numbers = [1]
    source_target_facets = {}
    dataset_split_facets = {}
    platform_summary = []
    source_target_summary = []
    quick_refine_links = []
    active_filters = []
    polarity_facets = {}
    subjectivity_facets = {}
    sentiment_summary = []
    dominant_sentiment = None
    
    try:
        # 4. Execute Search & Measure Speed (SC4021 Rubric Q2)
        start_time = time.time()
        results = solr.search(solr_query, fq=fq, **params)
        query_time = time.time() - start_time
        
        # 5. Parse Results
        num_found = results.hits
        total_pages = max(1, (num_found + rows_per_page - 1) // rows_per_page)
        if current_page > total_pages:
            current_page = total_pages
            params['start'] = (current_page - 1) * rows_per_page
            results = solr.search(solr_query, fq=fq, **params)
            num_found = results.hits
        docs = list(results)
        for doc in docs:
            doc["text"] = normalize_text_value(doc.get("text", ""))
            doc["platform"] = normalize_text_value(doc.get("platform", "Unknown"))
            doc["source_target"] = normalize_text_value(doc.get("source_target", "General"))
            doc["dataset_split"] = normalize_text_value(doc.get("dataset_split", ""))
            doc["subjectivity"] = normalize_text_value(doc.get("subjectivity", "")).lower()
            doc["polarity"] = normalize_text_value(doc.get("polarity", "")).lower()
            doc["highlighted_text"] = highlight_text(doc["text"], raw_query)
            doc["subjectivity_badge_class"] = sentiment_badge_class(doc["subjectivity"])
            doc["polarity_badge_class"] = sentiment_badge_class(doc["polarity"])
        page_numbers = get_page_window(current_page, total_pages)
        
        # Parse Solr facet arrays into a clean Python dictionary
        facets = parse_facet_counts(results, 'platform')
        source_target_facets = parse_facet_counts(results, 'source_target')
        dataset_split_facets = parse_facet_counts(results, 'dataset_split')
        polarity_facets = parse_facet_counts(results, 'polarity')
        subjectivity_facets = parse_facet_counts(results, 'subjectivity')
        platform_summary = build_summary_items(facets, num_found)
        source_target_summary = build_summary_items(source_target_facets, num_found)
        sentiment_summary = build_sentiment_summary(polarity_facets, num_found)
        dominant_sentiment = dominant_sentiment_from_facets(polarity_facets, num_found)
        quick_refine_links = build_quick_refine_links(
            raw_query,
            facets,
            source_target_facets,
            polarity_facets,
            platform_filter,
            source_target_filter,
            dataset_split_filter,
            polarity_filter,
            subjectivity_filter,
        )

        if platform_filter:
            active_filters.append({
                "label": f"Platform: {platform_filter}",
                "clear_url": build_search_url(
                    raw_query=raw_query,
                    platform='',
                    source_target=source_target_filter,
                    dataset_split=dataset_split_filter,
                    polarity=polarity_filter,
                    subjectivity=subjectivity_filter,
                ),
            })
        if source_target_filter:
            active_filters.append({
                "label": f"Target: {source_target_filter}",
                "clear_url": build_search_url(
                    raw_query=raw_query,
                    platform=platform_filter,
                    source_target='',
                    dataset_split=dataset_split_filter,
                    polarity=polarity_filter,
                    subjectivity=subjectivity_filter,
                ),
            })
        if dataset_split_filter:
            active_filters.append({
                "label": f"Split: {dataset_split_filter}",
                "clear_url": build_search_url(
                    raw_query=raw_query,
                    platform=platform_filter,
                    source_target=source_target_filter,
                    dataset_split='',
                    polarity=polarity_filter,
                    subjectivity=subjectivity_filter,
                ),
            })
        if polarity_filter:
            active_filters.append({
                "label": f"Polarity: {polarity_filter}",
                "clear_url": build_search_url(
                    raw_query=raw_query,
                    platform=platform_filter,
                    source_target=source_target_filter,
                    dataset_split=dataset_split_filter,
                    polarity='',
                    subjectivity=subjectivity_filter,
                ),
            })
        if subjectivity_filter:
            active_filters.append({
                "label": f"Subjectivity: {subjectivity_filter}",
                "clear_url": build_search_url(
                    raw_query=raw_query,
                    platform=platform_filter,
                    source_target=source_target_filter,
                    dataset_split=dataset_split_filter,
                    polarity=polarity_filter,
                    subjectivity='',
                ),
            })
            
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
        current_source_target=source_target_filter,
        current_dataset_split=dataset_split_filter,
        current_polarity=polarity_filter,
        current_subjectivity=subjectivity_filter,
        facets=facets,
        source_target_facets=source_target_facets,
        dataset_split_facets=dataset_split_facets,
        polarity_facets=polarity_facets,
        subjectivity_facets=subjectivity_facets,
        current_page=current_page,
        total_pages=total_pages,
        page_numbers=page_numbers,
        platform_summary=platform_summary,
        source_target_summary=source_target_summary,
        sentiment_summary=sentiment_summary,
        dominant_sentiment=dominant_sentiment,
        quick_refine_links=quick_refine_links,
        active_filters=active_filters,
        build_url=lambda **overrides: build_search_url(
            raw_query=overrides.get('raw_query', raw_query),
            platform=overrides.get('platform', platform_filter),
            source_target=overrides.get('source_target', source_target_filter),
            dataset_split=overrides.get('dataset_split', dataset_split_filter),
            polarity=overrides.get('polarity', polarity_filter),
            subjectivity=overrides.get('subjectivity', subjectivity_filter),
            page=overrides.get('page', 1),
        ),
    )

if __name__ == '__main__':
    # Run on port 5000. Debug mode auto-reloads the server when you make changes.
    app.run(host='0.0.0.0', port=8080, debug=True)
