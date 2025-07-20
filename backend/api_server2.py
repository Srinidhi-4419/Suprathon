from flask import Flask, request, jsonify
from flask_cors import CORS
from new_abc import analyze_with_groq, verify_with_gemini, combine_analysis
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze', methods=['POST'])
def abc_analyze():
    data = request.json
    if not data:
        return jsonify({'error': 'No input provided'}), 400
    url_or_text = data.get('url') or data.get('text')
    if not url_or_text:
        return jsonify({'error': 'No input provided'}), 400

    groq_result = analyze_with_groq(url_or_text)
    gemini_result = verify_with_gemini(groq_result)
    final_result = combine_analysis(groq_result, gemini_result)
    return jsonify(json.loads(final_result))

if __name__ == '__main__':
    app.run(port=5000)
