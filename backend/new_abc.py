import requests
import json
import re
from urllib.parse import urlparse

# === API KEYS ===
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# === ENHANCED GROQ ANALYSIS ===
def analyze_with_groq(url_or_text):
    prompt = f"""
You are an expert fact-checker with access to 1000+ verified news sources. Analyze the following news article with extreme precision:

{url_or_text}

CRITICAL INSTRUCTIONS FOR ACCURACY:
1. If the content contains multiple FAKE NEWS INDICATORS (sensational claims, no sources, emotional manipulation, conspiracy theories, misleading headlines), classify as FAKE
2. If from VERIFIED PUBLISHERS (BBC, Reuters, AP, NPR, PBS, major newspapers with .com/.org domains), lean toward REAL
3. Check for BIAS INDICATORS: loaded language, one-sided reporting, lack of expert quotes
4. Count EMOTIONAL MANIPULATION words: "shocking", "outrageous", "they don't want you to know", etc.
5. Look for EVIDENCE: citations, expert quotes, official sources, data

Perform these specific checks:
- Source credibility (check domain reputation)
- Writing quality and professionalism
- Presence of citations and sources
- Emotional manipulation tactics
- Factual consistency
- Publisher verification

Respond ONLY in this JSON format:
{{
    "title": "Article title",
    "author": "Author name or Unknown",
    "date": "Publication date or Unknown", 
    "summary": "2-3 line summary",
    "publisher": "Publisher name",
    "domain": "Domain name if URL provided",
    "credibility_score": 85,
    "reliability_level": "High/Medium/Low",
    "emotional_words_count": 3,
    "bias_indicators_count": 1,
    "linsear_write_formula": 12.4,
    "alexa_rank": 1500,
    "spam_detection_score": 15,
    "number_of_evidences": 5,
    "initial_judgment": "REAL/FAKE/UNCLEAR",
    "fake_indicators": ["list of fake news indicators found"],
    "credible_indicators": ["list of credibility indicators found"]
}}
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert fact-checker specializing in identifying fake news with 99% accuracy. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

# === ENHANCED GEMINI VERIFICATION ===
def verify_with_gemini(groq_analysis):
    # This is a placeholder. Replace with actual Gemini API call if available.
    # For now, return a mock response for demonstration.
    # You can integrate the Gemini API here as needed.
    prompt = f"""
You are a senior fact-checker with access to Snopes, PolitiFact, BBC Verify, Reuters Fact Check, and 500+ verification sources.

GROQ INITIAL ANALYSIS:
{groq_analysis}

ENHANCED VERIFICATION PROTOCOL:
1. Cross-reference claims against known fact-checking databases
2. Verify publisher authenticity and reputation
3. Check for known disinformation patterns
4. Validate evidence and source citations
5. Assess linguistic markers of fake news
6. Generate fact-check URLs for verification

ACCURACY REQUIREMENTS:
- If 3+ FAKE indicators present → classify as FAKE
- If from verified news sources with evidence → classify as REAL  
- If uncertain → provide specific reasoning

Generate fact-check URLs based on the content type and claims made.

Respond ONLY in this JSON format:
{{
    "credibility_score": 88,
    "reliability_level": "High",
    "emotional_words_count": 2,
    "bias_indicators_count": 1,
    "linsear_write_formula": 11.2,
    "alexa_rank": 950,
    "spam_detection_score": 8,
    "number_of_evidences": 7,
    "final_verdict": "REAL/FAKE",
    "confidence_level": 95,
    "fact_check_urls": [
        "https://www.snopes.com/fact-check/relevant-topic",
        "https://www.politifact.com/factchecks/relevant-check"
    ],
    "verification_reasoning": "Brief explanation of verdict",
    "red_flags": ["list of concerning elements"],
    "supporting_evidence": ["list of credible elements"]
}}
"""
    # Return a mock response for now
    return json.dumps({
        "credibility_score": 88,
        "reliability_level": "High",
        "emotional_words_count": 2,
        "bias_indicators_count": 1,
        "linsear_write_formula": 11.2,
        "alexa_rank": 950,
        "spam_detection_score": 8,
        "number_of_evidences": 7,
        "final_verdict": "REAL",
        "confidence_level": 95,
        "fact_check_urls": [
            "https://www.snopes.com/fact-check/relevant-topic",
            "https://www.politifact.com/factchecks/relevant-check"
        ],
        "verification_reasoning": "Brief explanation of verdict",
        "red_flags": ["list of concerning elements"],
        "supporting_evidence": ["list of credible elements"]
    })

# === COMBINE RESULTS ===
def combine_analysis(groq_result, gemini_result):
    try:
        # Clean JSON responses
        groq_data = json.loads(groq_result.strip().replace('```json', '').replace('```', ''))
        gemini_data = json.loads(gemini_result.strip().replace('```json', '').replace('```', ''))

        # Calculate weighted credibility score
        groq_score = groq_data.get('credibility_score', 50)
        gemini_score = gemini_data.get('credibility_score', 50)
        final_score = int((groq_score * 0.4) + (gemini_score * 0.6))  # Gemini weighted higher

        # Determine reliability level
        if final_score >= 80:
            reliability = "High"
        elif final_score >= 60:
            reliability = "Medium"
        else:
            reliability = "Low"

        # Final verdict logic
        groq_verdict = groq_data.get('initial_judgment', 'UNCLEAR')
        gemini_verdict = gemini_data.get('final_verdict', 'UNCLEAR')

        if groq_verdict == "FAKE" or gemini_verdict == "FAKE":
            final_verdict = "FAKE"
        elif groq_verdict == "REAL" and gemini_verdict == "REAL":
            final_verdict = "REAL"
        elif final_score >= 75:
            final_verdict = "REAL"
        elif final_score <= 40:
            final_verdict = "FAKE"
        else:
            final_verdict = "UNCLEAR"

        # Combined result
        combined_result = {
            "credibility_score": final_score,
            "reliability_level": reliability,
            "emotional_words_count": gemini_data.get('emotional_words_count', groq_data.get('emotional_words_count', 0)),
            "bias_indicators_count": gemini_data.get('bias_indicators_count', groq_data.get('bias_indicators_count', 0)),
            "linsear_write_formula": gemini_data.get('linsear_write_formula', groq_data.get('linsear_write_formula', 0.0)),
            "alexa_rank": gemini_data.get('alexa_rank', groq_data.get('alexa_rank', 999999)),
            "spam_detection_score": gemini_data.get('spam_detection_score', groq_data.get('spam_detection_score', 50)),
            "number_of_evidences": gemini_data.get('number_of_evidences', groq_data.get('number_of_evidences', 0)),
            "final_verdict": final_verdict,
            "fact_check_urls": gemini_data.get('fact_check_urls', []),
            "confidence_level": gemini_data.get('confidence_level', 70)
        }

        return json.dumps(combined_result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to parse JSON response: {str(e)}",
            "credibility_score": 0,
            "reliability_level": "Unknown",
            "final_verdict": "ERROR"
        }, indent=2)
