# NARAYANI: Grounded Emergency Dispatch Intelligence 🚑

Narayani is an enterprise-grade, real-time AI emergency dispatch system designed to reduce response times, increase dispatcher situational awareness, and provide immediate, empathetic medical guidance to callers before first responders arrive.

## 🌟 The Vision
When emergencies happen, seconds matter. Traditional 911/112 dispatch relies entirely on the caller's ability to verbally describe their situation, which can be compromised by panic, injury, or language barriers. 

Narayani bridges this gap by turning the caller's smartphone into an intelligent sensor suite:
1. **Live Video Grounding**: Real-time visual analysis of the scene, injuries, and environment.
2. **"Motherly" Conversational AI**: A deeply empathetic, multilingual voice AI that speaks native dialects and calms the caller while guiding them through life-saving first-aid.
3. **Instant Triage**: ML-powered severity scoring that fuses visual data, acoustic sentiment, and transcript analysis to automatically prioritize critical calls.
4. **Dispatcher Command Center**: A lag-free, real-time dashboard giving dispatchers superhuman situational awareness with live maps, transcripts, and AI-extracted summaries.

---

## 🏗️ System Architecture & Tech Stack

Narayani is built on a scalable, low-latency microservices architecture optimized for real-time streaming and high-availability.

### 1. Frontend Client (Caller & Dispatcher)
- **Core**: Vanilla HTML5, CSS3, JavaScript (ES6+). Zero heavy frontend frameworks to ensure the caller app loads instantly even on slow 3G networks.
- **Real-Time Communications**: `Socket.io` for bi-directional event streaming (WebSockets).
- **Media Acquisition**: `navigator.mediaDevices` for 1080p@60fps GPU-accelerated video streaming.
- **Mapping**: `Leaflet.js` for lightweight, custom-styled maps.

### 2. Backend API Server (Node.js)
- **Runtime**: Node.js & Express.
- **Real-Time Engine**: Socket.io server managing segregated rooms for callers and dispatchers.
- **Database**: MongoDB Atlas (Primary Datastore) & Redis (In-Memory Pub/Sub and caching for horizontal scaling).
- **Concurrency**: Asynchronous event loop optimized for handling concurrent HTTP API calls to LLMs and TTS/STT engines.

### 3. Machine Learning & Severity Engine (Python)
- **Runtime**: Python 3.10+ with Flask (Waitress WSGI for production).
- **Model**: Custom-trained `LightGBM` ensemble classifier.
- **Feature Extraction**: TF-IDF Vectorizer and `TextBlob` for sentiment polarity and subjectivity analysis.
- **Role**: Evaluates the linguistic structure, medical keywords, and emotional sentiment of the caller's text to assign a 1-10 severity score and classification (e.g., CARDIAC, TRAUMA).

### 4. Generative AI Pipeline
- **Reasoning Cascade**: `Groq` (Llama 3.1 8B Instant) as the primary ultra-fast reasoning engine, with an automatic fallback to `Mistral` (Mistral-Small-Latest) for high availability.
- **Speech-to-Text (STT)**: `Murf AI` for lightning-fast, highly accurate transcription of audio streams.
- **Text-to-Speech (TTS)**: `Murf AI` conversational models (`en-US-julia`, `hi-IN-ananya`) configured for slow, calm, and deeply empathetic intonation.

---

## 🧠 The AI Persona: "Narayani"

Narayani is not a robotic voice assistant. She is designed with the persona of a warm, loving, and experienced nurse or mother.

- **Context-Aware Memory**: The AI maintains a rolling window of the last 10 conversation turns, ensuring it never repeats itself and builds upon the caller's previous statements.
- **Multilingual Native Accents**: If a caller speaks Hindi (even in Roman script like *"mujhe dard ho raha hai"*), Narayani instantly switches not just her language, but her underlying TTS model to a native Indian voice (`hi-IN-ananya`), ensuring cultural comfort.
- **Zero-Hallucination Grounding**: The LLM is continuously fed "grounded frames" (descriptions of what the smartphone camera sees). If the caller says "my friend is bleeding," but the AI is told by the vision system that "a person is lying unconscious," the AI merges this physical truth with the verbal claim.

---

## 📊 ML Model & Datasets

### The Severity Classifier
The core triage engine is a `LightGBM` classifier trained to predict the severity of an emergency based on transcribed text.

- **Training Pipeline**: `scikit-learn` pipeline combining `TfidfVectorizer` (ngram_range=(1,2), max_features=3000) and `LGBMClassifier`.
- **Feature Engineering**: Extracted features include text length, exclamation marks, question marks, uppercase ratios, and TextBlob emotional polarity.
- **Labels**: Mapped to standard emergency triage colors (GREEN: Low, YELLOW: Moderate, ORANGE: High, RED: Critical).

### Datasets
The model was trained on a custom-curated, augmented dataset of over **8,500 emergency call transcripts**.
1. **Source 1**: Anonymized 911 dispatch logs (publicly available datasets).
2. **Source 2**: Synthetic generation of edge-cases (e.g., mixed English/Hindi "Hinglish" calls, panic-induced stuttering) using high-parameter LLMs to ensure robustness in the Indian sub-continent context.
3. **Data Augmentation**: NLP augmentation techniques (synonym replacement, random deletion) were applied to prevent overfitting on common phrases.

---

## 🚀 Setup & Installation

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- MongoDB Atlas account (with network access IP whitelisted)

### 1. Clone & Install
```bash
git clone https://github.com/your-username/narayani.git
cd narayani

# Install Node dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file in the root directory:
```env
# Server
PORT=3000
SEVERITY_SERVICE_URL=http://localhost:5050

# Database
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/narayani?retryWrites=true&w=majority
REDIS_URL=redis://localhost:6379 # Optional

# AI Providers
MURF_API_KEY=your_murf_key_here
GROQ_API_KEY=your_groq_key_here
MISTRAL_API_KEY=your_mistral_key_here
```
*(Note: Ensure your current IP is added to the MongoDB Atlas Network Access Whitelist).*

### 3. Start the Severity Engine (Python)
```bash
python severity_service.py
```
*Runs on port 5050 by default.*

### 4. Start the Main Server (Node.js)
```bash
node server.js
```
*Runs on port 3000 by default.*

---

## 🖥️ Usage

1. **Dispatcher Command Center**: Open `http://localhost:3000/dispatcher` on a large monitor.
2. **Caller Interface**: Open `http://localhost:3000/caller` (ideally on a mobile device or a separate browser window).
3. **Activate**: Click "Share Camera" and then tap the microphone button on the Caller UI to speak. The Dispatcher screen will instantly populate with live video, transcripts, AI summaries, and severity scores.

---

## 🚀 Deployment (Production)

**CRITICAL NOTE ON HOSTING:**
Because Narayani relies heavily on **WebSockets (Socket.io)** for real-time 60fps video streaming and live audio transcription, **it CANNOT be deployed on Serverless platforms like Vercel or Netlify.** Serverless functions immediately close persistent real-time connections.

You **MUST** deploy this on a platform that supports persistent Node.js servers. We recommend **Render.com** (it has a great free tier for Hackathons).

### How to deploy on Render (Free & Fast)
1. Go to [Render.com](https://render.com/) and create a free account.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`NARAYANI-AI`).
4. Configure the service:
   - **Environment:** Node
   - **Build Command:** `npm install`
   - **Start Command:** `node server.js`
5. Click **Advanced** and add your **Environment Variables**:
   - `MURF_API_KEY`=your_key
   - `GROQ_API_KEY`=your_key
   - `MISTRAL_API_KEY`=your_key
   - `MONGODB_URI`=mongodb+srv://...
6. Click **Create Web Service**. 

Within 2 minutes, your AI will be live with full real-time video/voice support!

---

## 🛡️ Production & Enterprise Readiness

- **Resilience**: The LLM pipeline uses an automatic cascade. If the primary provider (Groq) is rate-limited (HTTP 429) or fails, the system instantly and invisibly falls back to Mistral.
- **Graceful Degradation**: If Redis fails, the system safely falls back to single-node memory mapping. If MongoDB is slow, critical real-time features continue to operate via Socket.io.
- **Resource Optimization**: GPU-accelerated CSS `will-change` properties and optimal `100ms` frame-sampling rates ensure the dispatcher dashboard runs smoothly at 60fps without causing browser memory leaks.
- **Data Security**: All API keys are strictly managed server-side. No sensitive tokens are exposed to the browser client.



By G R Harsha

---
*Built to save lives, one call at a time.*
