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
        .quick-filter-link { display: inline-flex; align-items: center; border: 1px solid #ced4da; border-radius: 999px; padding: 7px 12px; margin: 0 10px 10px 0; text-decoration: none; color: #212529; background: #fff; transition: all 0.15s ease; font-size: 0.92rem; }
        .quick-filter-link:hover { background: #f1f5f9; border-color: #0d6efd; color: #0d6efd; }
        .filter-chip { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; background: #212529; color: #fff; padding: 6px 10px; margin: 0 8px 8px 0; font-size: 0.9rem; }
        .filter-chip a { color: #fff; text-decoration: none; font-weight: 700; }
        .viz-toolbar { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
        .viz-select { min-width: 220px; }
        .insight-band { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; padding: 14px 16px; border-radius: 18px; border: 1px solid #dbe7f5; }
        .insight-positive { background: linear-gradient(135deg, #ecfdf3 0%, #f7fff9 100%); border-color: #b7ebcb; }
        .insight-negative { background: linear-gradient(135deg, #fff1f1 0%, #fff9f9 100%); border-color: #f1b8b8; }
        .insight-neutral { background: linear-gradient(135deg, #f1f3f5 0%, #fafbfc 100%); border-color: #d6dbe1; }
        .insight-copy strong { display: block; font-size: 1rem; color: #12263a; }
        .insight-copy span { color: #5b6776; font-size: 0.93rem; }
        .insight-metrics { display: flex; gap: 10px; flex-wrap: wrap; }
        .insight-pill { border-radius: 999px; padding: 8px 12px; background: #fff; border: 1px solid #d6e0eb; font-size: 0.9rem; color: #334155; }
        .word-cloud-canvas { position: relative; min-height: 420px; border-radius: 34px; overflow: hidden; background:
            radial-gradient(circle at 28% 56%, rgba(255,255,255,0.98) 0, rgba(255,255,255,0.98) 28%, rgba(255,255,255,0) 29%),
            radial-gradient(circle at 47% 34%, rgba(255,255,255,0.98) 0, rgba(255,255,255,0.98) 30%, rgba(255,255,255,0) 31%),
            radial-gradient(circle at 66% 55%, rgba(255,255,255,0.98) 0, rgba(255,255,255,0.98) 28%, rgba(255,255,255,0) 29%),
            radial-gradient(circle at 50% 68%, rgba(255,255,255,0.98) 0, rgba(255,255,255,0.98) 32%, rgba(255,255,255,0) 33%),
            linear-gradient(180deg, #dff3ff 0%, #eef9ff 58%, #f8fcff 100%);
            box-shadow: inset 0 -20px 40px rgba(13, 110, 253, 0.06); }
        .word-cloud-canvas::after { content: ""; position: absolute; inset: auto 6% 8% 6%; height: 18%; border-radius: 999px; background: rgba(190, 230, 255, 0.45); filter: blur(18px); }
        .word-cloud-term { position: absolute; text-decoration: none; line-height: 1; color: #1f2937; transform: translate(-50%, -50%) rotate(var(--rotate, 0deg)); white-space: nowrap; text-shadow: 0 1px 0 rgba(255,255,255,0.85); }
        .word-cloud-term:hover { color: #0d6efd; }
        .word-cloud-rank-1 { font-size: 2.2rem; font-weight: 700; }
        .word-cloud-rank-2 { font-size: 1.9rem; font-weight: 700; }
        .word-cloud-rank-3 { font-size: 1.6rem; font-weight: 600; }
        .word-cloud-rank-4 { font-size: 1.35rem; font-weight: 600; }
        .word-cloud-rank-5 { font-size: 1.15rem; font-weight: 500; }
        .viz-description { font-size: 0.92rem; color: #6c757d; margin-bottom: 0; }
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
        {% if current_viz %}
            <input type="hidden" name="viz" value="{{ current_viz }}">
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
                            <div class="card-header bg-light">Visualization Explorer</div>
                            <div class="card-body">
                                {% if dominant_sentiment %}
                                    <div class="insight-band insight-{{ dominant_sentiment.tone_class }}">
                                        <div class="insight-copy">
                                            <strong>{{ dominant_sentiment.label }} sentiment is leading this conversation.</strong>
                                            <span>{{ dominant_sentiment.percent }}% of the current matches lean {{ dominant_sentiment.label.lower() }}. Use the sentiment facet or the badges below to drill into the opinions driving that trend.</span>
                                        </div>
                                        <div class="insight-metrics">
                                            {% for item in sentiment_summary %}
                                                <span class="insight-pill">{{ item.label }}: {{ item.count }}</span>
                                            {% endfor %}
                                        </div>
                                    </div>
                                {% endif %}
                                <div class="viz-toolbar">
                                    <div>
                                        <p class="viz-description">Choose how to read the current result set: as a thematic cloud, as a platform mix, or as a target mix.</p>
                                    </div>
                                    <form method="GET" action="/">
                                        <input type="hidden" name="q" value="{{ current_query }}">
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
                                        <label class="form-label mb-1">Visualization</label>
                                        <select class="form-select viz-select" name="viz" onchange="this.form.submit()">
                                            {% for option in visualization_options %}
                                                <option value="{{ option.value }}" {% if option.value == current_viz %}selected{% endif %}>{{ option.label }}</option>
                                            {% endfor %}
                                        </select>
                                    </form>
                                </div>

                                {% if current_viz == 'platform' %}
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
                                {% elif current_viz == 'target' %}
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
                                {% elif current_viz == 'wordcloud' %}
                                    {% if word_cloud_terms %}
                                        <div class="word-cloud-canvas">
                                            {% for item in word_cloud_terms %}
                                                <a
                                                    class="word-cloud-term word-cloud-rank-{{ item.rank }}"
                                                    href="{{ build_url(raw_query=item.query) }}"
                                                    style="left: {{ item.x }}%; top: {{ item.y }}%; --rotate: {{ item.rotate }}deg;"
                                                >{{ item.term }}</a>
                                            {% endfor %}
                                        </div>
                                        <p class="viz-description mt-3">The cloud surfaces the strongest themes inside the current search results. Click any term to refine the search toward that theme.</p>
                                    {% else %}
                                        <p class="text-muted mb-0">Not enough text to generate a word cloud for this query.</p>
                                    {% endif %}
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>

                {% if quick_refine_links %}
                    <div class="card mb-4 shadow-sm interactive-card">
                        <div class="card-header bg-light">Quick Filters</div>
                        <div class="card-body">
                            {% for item in quick_refine_links %}
                                <a class="quick-filter-link" href="{{ item.url }}">{{ item.label }}</a>
                            {% endfor %}
                        </div>
                    </div>
                {% elif num_found is not none %}
                    <div class="card mb-4 shadow-sm interactive-card">
                        <div class="card-header bg-light">Quick Filters</div>
                        <div class="card-body">
                            <p class="text-muted mb-0">No additional quick filters available for the current selection.</p>
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


def build_search_url(raw_query="", platform="", source_target="", dataset_split="", polarity="", subjectivity="", viz="", page=1):
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
    if viz:
        params["viz"] = viz
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


def build_query_refine_text(raw_query, term):
    if not raw_query:
        return term
    if term.lower() in raw_query.lower():
        return raw_query
    return f"{raw_query} {term}".strip()


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
        "tone_class": label if label in {"positive", "negative", "neutral"} else "neutral",
    }


def build_word_cloud_terms(docs, raw_query, limit=18):
    stop_words = {
        "the", "and", "for", "that", "this", "with", "have", "has", "had", "are", "was", "were",
        "but", "not", "you", "your", "from", "they", "them", "their", "its", "it's", "just",
        "about", "into", "over", "under", "more", "than", "very", "much", "can", "could", "would",
        "should", "will", "there", "here", "what", "when", "where", "which", "while", "then",
        "also", "because", "really", "still", "only", "being", "been", "even", "some", "like",
        "electric", "vehicle", "vehicles", "car", "cars", "ev", "one", "get", "got", "too", "our",
        "out", "all", "how", "why", "who", "his", "her", "she", "him", "his", "her", "than"
    }
    query_terms = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9']+", raw_query)
        if len(token) >= 3
    }

    counts = {}
    for doc in docs:
        text = normalize_text_value(doc.get("text", "")).lower()
        for token in re.findall(r"[a-zA-Z0-9']+", text):
            if len(token) < 3:
                continue
            if token in stop_words or token in query_terms:
                continue
            counts[token] = counts.get(token, 0) + 1

    top_terms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    if not top_terms:
        return []

    positions = [
        (50, 28, -2), (34, 40, -7), (66, 40, 6), (50, 50, 0), (24, 54, -5), (76, 54, 4),
        (38, 62, 3), (62, 62, -4), (50, 71, 2), (19, 40, -8), (81, 40, 8), (30, 28, -6),
        (70, 28, 7), (15, 58, -3), (85, 58, 3), (40, 78, -2), (60, 78, 2), (50, 84, 0),
    ]
    ranks = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]
    items = []
    for idx, (term, count) in enumerate(top_terms):
        rank = ranks[idx] if idx < len(ranks) else 5
        x, y, rotate = positions[idx] if idx < len(positions) else (50, 50, 0)
        items.append({
            "term": term,
            "count": count,
            "rank": rank,
            "query": build_query_refine_text(raw_query, term),
            "x": x,
            "y": y,
            "rotate": rotate,
        })
    return items


def build_quick_refine_links(raw_query, platform_facets, source_target_facets, polarity_facets, current_platform, current_source_target, current_dataset_split, current_polarity, current_subjectivity):
    links = []

    if not current_platform:
        for platform in list(platform_facets.keys())[:3]:
            links.append({
                "label": f"Platform: {platform} ({platform_facets[platform]})",
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
            links.append({
                "label": f"Target: {source_target} ({source_target_facets[source_target]})",
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
            links.append({
                "label": f"Sentiment: {polarity.title()} ({polarity_facets[polarity]})",
                "url": build_search_url(
                    raw_query=raw_query,
                    platform=current_platform,
                    source_target=current_source_target,
                    dataset_split=current_dataset_split,
                    polarity=polarity,
                    subjectivity=current_subjectivity,
                ),
            })

    return links[:8]

@app.route('/', methods=['GET'])
def index():
    # 1. Capture inputs
    raw_query = request.args.get('q', '').strip()
    platform_filter = request.args.get('platform', '')
    source_target_filter = request.args.get('source_target', '').strip()
    dataset_split_filter = request.args.get('dataset_split', '').strip()
    polarity_filter = request.args.get('polarity', '').strip().lower()
    subjectivity_filter = request.args.get('subjectivity', '').strip().lower()
    visualization = request.args.get('viz', 'wordcloud').strip().lower() or 'wordcloud'
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
    word_cloud_terms = []
    visualization_options = [
        {"value": "wordcloud", "label": "Word Cloud"},
        {"value": "platform", "label": "Platform Mix"},
        {"value": "target", "label": "Source/Target Mix"},
    ]
    valid_visualizations = {option["value"] for option in visualization_options}
    if visualization not in valid_visualizations:
        visualization = "wordcloud"
    
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
        word_cloud_terms = build_word_cloud_terms(docs, raw_query)
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
        current_viz=visualization,
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
        word_cloud_terms=word_cloud_terms,
        visualization_options=visualization_options,
        quick_refine_links=quick_refine_links,
        active_filters=active_filters,
        build_url=lambda **overrides: build_search_url(
            raw_query=overrides.get('raw_query', raw_query),
            platform=overrides.get('platform', platform_filter),
            source_target=overrides.get('source_target', source_target_filter),
            dataset_split=overrides.get('dataset_split', dataset_split_filter),
            polarity=overrides.get('polarity', polarity_filter),
            subjectivity=overrides.get('subjectivity', subjectivity_filter),
            viz=overrides.get('viz', visualization),
            page=overrides.get('page', 1),
        ),
    )

if __name__ == '__main__':
    # Run on port 5000. Debug mode auto-reloads the server when you make changes.
    app.run(host='0.0.0.0', port=8080, debug=True)
