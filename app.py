from flask import Flask, request, jsonify
from flask_cors import CORS
import pysolr

app = Flask(__name__)
# Enable CORS so your future frontend (HTML/JS) can talk to this API
CORS(app) 

# Connect to the Solr database
SOLR_URL = 'http://localhost:8983/solr/opinion_engine'
solr = pysolr.Solr(SOLR_URL, always_commit=True)

@app.route('/api/search', methods=['GET'])
def search():
    """
    Endpoint to handle search queries from the frontend.
    Example Usage: http://localhost:5000/api/search?q=battery&source=Reddit
    """
    # 1. Get query parameters from the user's request
    user_query = request.args.get('q', '*:*') # Default to all documents if empty
    source_filter = request.args.get('source', None) # e.g., 'Reddit' or 'X'
    
    # 2. Build the Solr query
    # Solr uses Lucene query syntax. If user types "battery", we search the "text" field.
    if user_query != '*:*':
        solr_query = f"text:({user_query})"
    else:
        solr_query = "*:*"

    # 3. Apply filters (if the user selected a specific platform)
    # fq stands for "Filter Query" in Solr
    filter_queries =[]
    if source_filter:
        filter_queries.append(f"source:{source_filter}")

    try:
        # 4. Execute the search against Solr
        # We fetch up to 50 results. 'fq' applies our source filters.
        results = solr.search(q=solr_query, fq=filter_queries, rows=50)
        
        # 5. Format the results to send back to the frontend
        formatted_results =[]
        for doc in results:
            formatted_results.append({
                "id": doc.get("id"),
                "source": doc.get("source", ["Unknown"])[0], # Solr sometimes returns lists
                "text": doc.get("text", [""])[0],
                "date_posted": doc.get("date_posted", [""])[0],
                "engagement_score": doc.get("engagement_score", [0])[0],
                "author": doc.get("author",["Unknown"])[0]
            })

        return jsonify({
            "status": "success",
            "total_found": results.hits, # How many total matches Solr found
            "results": formatted_results
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)