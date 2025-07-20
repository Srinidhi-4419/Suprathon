# 📰 TruthLens AI - News Credibility Checker

*TruthLens AI* is a powerful, AI-driven web platform that verifies the authenticity of news articles. It uses large language models, semantic analysis, and external fact-checking APIs to detect misinformation, evaluate bias, and provide transparent credibility scoring. Whether you're verifying a breaking headline or investigating suspicious content, TruthLens AI gives you clear, evidence-backed insights.

---

## 🌟 Website Description

TruthLens AI is a modern, single-page web application designed with *Next.js, offering an immersive 3D interface and real-time analytics dashboard. Users can submit either a **news URL* or *paste the full text, then select between **Fast Learner* and *Deep Learner* modes for credibility checking. The backend is powered by a Flask API integrated with Groq, Gemini, and custom NLP models.

---

## ⚙ Modes of Analysis

TruthLens AI offers two powerful modes tailored to user needs:

### 🚀 Fast Learner (LLM-Based Mode)
- *Powered by*: Groq LLM (Llama3-8b-8192) + Google Gemini
- *Input Supported*: URL or plain text
- *Speed*: ⚡ Ultra-fast (under 1 second)
- *Purpose*:
  - Rapid article screening
  - LLM-based judgment on tone, credibility, bias, and citation presence
- *Use Case*: Ideal for quick social media verification or first-pass analysis

---

### 🧠 Deep Learner (Multi-Model Verification Mode)
- *Powered by*: Custom AI/NLP models + Groq + Gemini + Fact-Check APIs
- *Input Supported*: URL or plain text
- *Speed*: ⏱ 2–5 seconds
- *Purpose*:
  - Scrapes full content
  - Extracts key claims
  - Performs multi-source fact-checking and semantic similarity
  - Detects emotion, subjectivity, stylometry, and writing complexity
- *Use Case*: Best for journalists, researchers, or deep investigative analysis

---

🎯 *Both Modes Support*:
- ✅ *URL Input*: Automatically scrapes and parses online articles
- ✍ *Text Input*: Manually verify custom-written or copied text

---

## 🚀 Key Features

- 🎯 *Credibility Score (0–100)*: Overall reliability index
- 🧠 *Multi-Layer AI Verification Pipeline*
  - Groq LLM for initial review
  - Gemini for secondary verification
  - NLP heuristics and semantic models
- 📊 *Interactive Visualizations*
  - Truth-O-Meter (gauge chart)
  - Source network graphs (D3.js)
  - Emotion & bias heatmap
  - Timeline of fact-checks and claim evidence
- 🔍 *Real-Time Fact Checking*
  - Evidence-based claim validation
  - Fact-check links with matched snippets
- 🧾 *Technical & Emotional Analysis*
  - Bias indicator
  - Emotion word count
  - Grade-level readability
  - Spam detection

---

## 🛠 Technology Stack

### 🔷 Frontend
- Next.js (15.4.2) — App framework
- React (19.1.0) — Component system
- Tailwind CSS (v4) — Styling utility
- Three.js — 3D background particle effects
- Framer Motion — Animations & transitions
- Recharts, D3.js — Charts and network graphs
- React Typewriter — UI enhancement for text effects

### 🔶 Backend
- Python 3.x + Flask 2.3.3 — API engine
- Flask-CORS — Cross-origin support
- Gunicorn / Uvicorn — For deployment (optional)

---

## 🤖 AI & NLP Components

### 🔹 LLM APIs
- *Groq API* — Llama3-8b-8192 (primary judgment model)
- *Google Gemini API* — Gemini 1.5 Flash 8B (cross-verification)

### 🔹 Custom NLP & ML
- *Sentence Transformers* — All-MiniLM-L6-v2 for semantic matching
- *NLTK, **TextBlob, **SpaCy* — NLP, bias, emotion & grammar
- *Scikit-learn* — ML logic for heuristics & scoring

### 🔹 Data Extraction
- Newspaper3k — News content parsing
- BeautifulSoup — HTML cleanup
- TextStat, NumPy — Complexity, math & stats
- Python-Whois — Domain trustworthiness

---

## 🔗 External APIs & Verification Sources

### 🧠 Fact-Checking APIs
- *Google Fact Check API* — Official claim verifications
- *Wikipedia/Wikidata APIs* — Structured data & summary extraction
- *News Aggregators*: 
  - NewsAPI.org
  - GNews.io
  - MediaStack

### 🕵‍♂ Fact-Check Platforms (via scraping)
- Snopes, PolitiFact, FactCheck.org, Lead Stories, AP, Reuters, BBC Verify, AFP, Full Fact, USA Today and others

---

## 📈 Output Metrics & Fields

Your API response returns detailed fields:
- credibility_score (0–100)
- source_reliability (0–100)
- bias_indicator_count
- emotion_word_count
- factcheck_links[]
- evidence_count
- grade_level
- spam_score
- popularity_rank
- confidence_level
- final_verdict: REAL, UNCLEAR, or FAKE

---

## 🧠 Final Verdict Classification

| Verdict | Description |
|--------|-------------|
| ✅ *REAL* | Verified with high evidence & consistency |
| ⚠ *UNCLEAR* | Lacks sufficient or reliable evidence |
| ❌ *FAKE* | Strong indicators of misinformation or bias |

---

## 🎯 Use Cases

- 📰 *Journalists*: Pre-publish verification and claim sourcing
- 🧑‍🏫 *Educators*: Teach news literacy and digital awareness
- 👥 *Public*: Verify breaking news or viral claims
- 🕵‍♀ *Researchers*: Analyze trends in misinformation

---

## 🔐 Privacy & Ethics

- ❌ No content stored after analysis
- 🔒 All communication uses HTTPS
- 🔑 API keys stored securely in server
- 💡 Transparent explanations for each score component

---

*TruthLens AI* empowers people with facts — cutting through misinformation with clarity, speed, and trust.  
🔍 Stay informed. Stay critical. Stay true.
