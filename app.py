from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from navigator import NavigatorAgent
from scraper import SeedToScaleScraper
from vectorizer import Vectorizer
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

# # Initialize the navigator agent
# navigator = NavigatorAgent()

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    k = data.get('k', 5)  # Number of results to return
    results = navigator.search(query, k=k)
    
    return jsonify({'results': results})

@app.route('/api/answer', methods=['POST'])
def answer():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    answer_data = navigator.answer_question(query)
    
    return jsonify(answer_data)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting to scrape SeedToScale...")
    scraper = SeedToScaleScraper()
    scraper.scrape_all_content()
    print("Creating vector embeddings...")
    vectorizer = Vectorizer()
    navigator = NavigatorAgent()
    vectorizer.create_embeddings()
    app.run(host='0.0.0.0', port=port, debug=True)