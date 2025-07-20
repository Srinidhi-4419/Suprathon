from flask import Flask, request, jsonify
from news_authenticity_checker import run_checker
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins for development

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    text = data.get('text')
    result = run_checker(url=url, text=text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5001) 