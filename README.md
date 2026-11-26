# 🎓 AI-Powered Smart Study Assistant (SmartStudy.ai)

> An intelligent, local AI study companion that transforms raw study materials into structured summaries, contextual conversational Q&A, interactive self-assessment quizzes, and comprehensive learning analytics.

[![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/Flask-3.1-black.svg)](https://palletsprojects.com/p/flask/)
[![AI Models](https://img.shields.io/badge/AI-Transformers%20%7C%20Sentence--Transformers%20%7C%20PyTorch-orange.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-100%2B%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 🌟 Key Capabilities by Development Phase

SmartStudy.ai was developed through 6 rigorous engineering phases, fully integrated into a unified full-stack architecture:

### 1. ⚙️ Phase 1 – Environment & Scaffolding
- Robust Flask 3.1 application factory (`create_app`) with multi-environment configuration (`Development`, `Testing`, `Production`).
- Automated health diagnostic subsystem (`/health`) monitoring database connectivity, upload paths, and AI model states.
- Responsive dark-mode glassmorphism design system (`style.css`) with micro-animations and mobile accessibility.

### 2. 📄 Phase 2 – PDF Processing & Extraction Engine
- **PyMuPDF (`fitz`) Integration:** High-speed document text and layout extraction with sub-page coordinate tracking.
- **Validation Pipeline:** Detection and rejection of corrupt files, zero-byte uploads, password-encrypted PDFs, and invalid headers.
- **Scanned Document Detection:** Heuristic text-density checks warning users if a document is an image scan requiring OCR.
- **Relational Document Store:** SQLite persistence storing file hashes, page counts, raw text, and metadata.

### 3. 🧠 Phase 3 – AI-Powered Study Summarization
- **Abstractive Summarization:** Powered by Hugging Face Transformers (`sshleifer/distilbart-cnn-12-6`).
- **Multi-Tier Summary Modes:**
  - *Quick Summary:* High-level bulleted digest.
  - *Standard Summary:* Balanced conceptual breakdown.
  - *Detailed Summary:* In-depth structural review with sections.
- **Intelligent Chunking:** Recursive map-reduce chunking preserving sentence boundaries for long documents.
- **Key Concept & Keyword Extraction:** TF-IDF n-gram extraction combined with POS-tagged concept identification and exam focus points.

### 4. 💬 Phase 4 – RAG-Based Contextual Chatbot
- **Document Chunking & Vectorization:** Text splitting with customizable overlap and page tracking.
- **Semantic Search Engine:** Dense vector embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` with FAISS-CPU vector indexing.
- **Contextual Grounding:** Top-K similarity retrieval injecting precise document context into `google/flan-t5-small`.
- **Anti-Hallucination Guardrails:** Strict confidence filtering preventing out-of-context answers and attributing page numbers.
- **Interactive Chat Interface:** Multi-turn conversational UI with markdown rendering, source citations, suggested questions, and session management.

### 5. 📝 Phase 5 – AI Quiz Generation & Interactive Assessment
- **Automated MCQ Generation:** Extracts salient facts, definitions, and key concepts from document text to produce 4-option multiple choice questions.
- **Plausible Distractor Synthesis:** Semantic domain matching ensuring distractors are meaningful and educational.
- **Adaptive Difficulty Levels:**
  - *Easy:* Direct factual recall and vocabulary identification.
  - *Medium:* Conceptual understanding and applied definitions.
  - *Hard:* Complex relationships, multi-step inference, and negative-case questions.
- **Secure Server-Side Grading:** Client payload scrubbed of answer keys; authoritative evaluation on submission.
- **Interactive Quiz Interface:** Real-time countdown timer, progress bar, flag question feature, and immediate score breakdowns with detailed explanations.
- **Attempt History & Analytics:** Historical quiz storage, score trends, and difficulty distribution tracking.

### 6. 📊 Phase 6 – Student Dashboard, Progress Analytics & Goals
- **Comprehensive Student Dashboard (`/dashboard`):** Real-time database metrics displaying total uploads, summaries, chat interactions, quizzes completed, average score, overall progress %, and study sessions.
- **Transparent 5-Pillar Progress Index:** Mathematical progress formula (0–100%) grounded entirely in verified student activities:
  - Document Uploads (20 pts)
  - AI Summaries (20 pts)
  - Chatbot Inquiries (20 pts)
  - Quizzes Attempted (20 pts)
  - Quiz Mastery & Accuracy (20 pts)
- **Performance Analytics (`/analytics`):**
  - Weekly study activity distribution (Mon–Sun bar chart).
  - Quiz score trajectory over time (chronological timeline chart).
  - Difficulty breakdown (Easy, Medium, Hard pass rates & averages).
  - Comprehensive learning stats grid (accuracy, total questions answered, highest/lowest scores).
- **Study Goals Manager (`/goals`):** Create, track, auto-sync, and manage custom study targets (e.g., target quizzes, summaries, or score thresholds).
- **Activity History Log (`/activity`):** Chronological activity timeline with multi-type and date-range filters.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Backend Framework** | Python 3.14+, Flask 3.1, Jinja2 |
| **Database** | SQLite3 with WAL mode, Foreign Keys & Cascading Deletes |
| **PDF Processing** | PyMuPDF (`fitz`) |
| **AI / NLP Models** | Hugging Face Transformers, Sentence-Transformers, PyTorch |
| **Vector Search** | FAISS-CPU (FlatIP inner product with L2 normalized vectors) |
| **Text Analytics** | Scikit-learn (TF-IDF vectorizer), NumPy |
| **Frontend UI** | HTML5, CSS3 Glassmorphism, Vanilla ES6+ JavaScript, Responsive SVG Charts |
| **Testing & QA** | Pytest, Pytest-cov, Functional HTTP E2E Test Suites |

---

## 📁 Project Directory Structure

```text
AI-Powered Smart Study Assistant/
├── app/
│   ├── __init__.py               # Flask application factory (create_app)
│   ├── routes.py                 # Application routes (Phases 1-6 integrated)
│   ├── summarizer.py             # Summarizer facade re-export
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Unified glassmorphism stylesheet & responsive system
│   │   └── js/
│   │       └── script.js         # Frontend interactions, AJAX, timers, modal controls
│   └── templates/
│       ├── base.html             # Base layout template with navigation & notification toasts
│       ├── dashboard.html        # Phase 6: Student Dashboard (10 interactive sections)
│       ├── analytics.html        # Phase 6: Performance Analytics & Visual Charts
│       ├── goals.html            # Phase 6: Study Goals Management & Progress Sync
│       ├── activity.html         # Phase 6: Chronological Activity Log & Filter System
│       ├── index.html            # Upload Center & PDF onboarding
│       ├── documents.html        # Document library index
│       ├── document_details.html # Document viewer, tabs & AI Study Summarizer
│       ├── chatbot.html          # Phase 4: RAG-based AI Chatbot
│       ├── quiz_hub.html         # Phase 5: Quizzes Hub & Document Selector
│       ├── quiz_config.html      # Phase 5: Quiz Generation Configuration
│       ├── quiz_take.html        # Phase 5: Interactive Quiz Player with Timer
│       ├── quiz_results.html     # Phase 5: Quiz Result Breakdown & Explanations
│       └── quiz_history.html     # Phase 5: Quiz Attempts History & Review
│
├── database/
│   ├── __init__.py
│   ├── db.py                     # SQLite schema definitions, CRUD, migrations, analytics queries
│   └── smart_study.db            # Active SQLite database file
│
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py          # PyMuPDF text & metadata extraction, file validation
│   ├── summarizer.py             # Hugging Face DistilBART summarization engine
│   ├── text_chunker.py           # Recursive chunking with overlap & sentence boundaries
│   ├── embedding_engine.py       # Sentence-Transformers vector embedding generation
│   ├── vector_store.py           # FAISS vector database store & similarity search
│   ├── rag_qa.py                 # RAG pipeline & Flan-T5 contextual question-answering
│   └── quiz_generator.py         # AI question synthesis, distractor generation & grading
│
├── tests/
│   ├── __init__.py
│   ├── test_basic.py             # Phase 1: Basic routes & health endpoint
│   ├── test_pdf_processing.py    # Phase 2: PDF extraction & validation tests
│   ├── test_summarizer.py        # Phase 3: Summarization, keyword & chunking tests
│   ├── test_chatbot.py           # Phase 4: Embedding, FAISS, RAG & Chatbot tests
│   ├── test_quiz_generator.py    # Phase 5: Question generation, distractor & grading tests
│   ├── test_quiz_routes.py       # Phase 5: Quiz route handlers & security tests
│   ├── test_dashboard_analytics.py # Phase 6: Dashboard, stats, analytics & goals tests
│   ├── functional_test_phase4.py # Phase 4 Live E2E functional test script
│   ├── functional_test_phase5.py # Phase 5 Live E2E functional test script
│   └── functional_test_phase6.py # Phase 6 Live E2E functional test script
│
├── models/
│   └── cache/                    # Local Hugging Face & Sentence-Transformers model cache
├── uploads/                      # User-uploaded PDF document storage
├── config.py                     # Application configuration (Dev, Test, Prod)
├── run.py                        # Application entry point script
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Tested and verified on **Python 3.14.2**)
- **Git**

### 2. Clone Repository
```powershell
git clone <repository-url>
cd "AI-Powered Smart Study Assistant"
```

### 3. Create & Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Run the Application
```powershell
python run.py
```

The application will start on:
- **Student Dashboard:** [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Document Library:** [http://127.0.0.1:5000/documents](http://127.0.0.1:5000/documents)
- **Quizzes Hub:** [http://127.0.0.1:5000/quizzes](http://127.0.0.1:5000/quizzes)
- **Performance Analytics:** [http://127.0.0.1:5000/analytics](http://127.0.0.1:5000/analytics)
- **Study Goals:** [http://127.0.0.1:5000/goals](http://127.0.0.1:5000/goals)
- **Activity Log:** [http://127.0.0.1:5000/activity](http://127.0.0.1:5000/activity)
- **System Health Diagnostics:** [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

---

## 🩺 Health Check & System Verification

The backend exposes a comprehensive health diagnostic API endpoint at `/health`:
```bash
curl http://127.0.0.1:5000/health
```

Sample JSON response:
```json
{
  "status": "healthy",
  "service": "AI-Powered Smart Study Assistant",
  "version": "1.0.0",
  "phase": "Phases 1-6 Integrated (Phase 3 Summarization, Phase 4 Chatbot RAG, Phase 5 AI Quizzes, Phase 6 Dashboard & Analytics)",
  "database_ready": true,
  "upload_folder_ready": true,
  "rag_embedding_model": "all-MiniLM-L6-v2",
  "rag_qa_model": "google/flan-t5-small",
  "quiz_generator": "ready",
  "quiz_evaluator": "ready",
  "dashboard_analytics": "ready",
  "study_goals": "ready",
  "activity_tracker": "ready"
}
```

---

## 🧪 Automated Testing Suite

The project includes an extensive automated test suite covering all 6 phases:

### Running Unit & Integration Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Running Phase-Specific Tests:
```powershell
# Phase 6: Dashboard, Progress Tracking & Analytics
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_analytics.py -v

# Phase 5: AI Quiz Generation & Assessment
.\.venv\Scripts\python.exe -m pytest tests/test_quiz_generator.py tests/test_quiz_routes.py -v

# Phase 4: RAG Vector Store & Contextual Chatbot
.\.venv\Scripts\python.exe -m pytest tests/test_chatbot.py -v

# Phase 3: AI Summarization Engine
.\.venv\Scripts\python.exe -m pytest tests/test_summarizer.py -k "not test_real_distilbart" -v

# Phase 2: PDF Parsing & Validation
.\.venv\Scripts\python.exe -m pytest tests/test_pdf_processing.py -v
```

### Running Live Functional Verification:
```powershell
# Start server in one terminal:
python run.py

# In another terminal:
python tests/functional_test_phase6.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` or `GET /dashboard` | GET | Render Student Dashboard view or fetch stats JSON |
| `GET /api/dashboard/stats` | GET | Real-time dashboard statistics and progress index |
| `GET /analytics` | GET | Render Performance Analytics page |
| `GET /api/analytics/data` | GET | Weekly distribution, score timeline, difficulty metrics |
| `GET /goals` | GET | Render Study Goals management page |
| `GET /api/goals` | GET | List active and completed study goals with synced progress |
| `POST /api/goals/create` | POST | Create a new study goal |
| `POST /api/goals/<id>/update` | POST | Update goal target, progress, or status |
| `POST /api/goals/<id>/delete` | POST | Delete study goal |
| `GET /activity` | GET | Render Activity History timeline view |
| `GET /api/activity/history` | GET | Filterable study activity history (type, date, pagination) |
| `POST /upload` | POST | Upload and process PDF document |
| `POST /document/<id>/summarize`| POST | Trigger AI summarization (quick, standard, detailed) |
| `POST /api/chat/<id>/ask` | POST | Contextual RAG question answering |
| `POST /document/<id>/quiz/generate` | POST | Generate AI MCQ quiz from document |
| `POST /quiz/<id>/submit` | POST | Authoritative server-side quiz grading |
| `GET /health` | GET | Real-time system health and module status |

---

## 🔒 Security & Best Practices
- **No Client-Side Answer Leakage:** Quiz questions returned to browser never contain answer keys or rationale.
- **Strict Anti-Hallucination:** Chatbot questions ungrounded in document context return polite fallback notices without fabricating facts.
- **Zero-State Resilience:** All dashboard, analytics, and history components cleanly render informative zero-states on fresh databases.
- **Relational Integrity:** SQLite foreign keys with cascading deletions ensure document removal purges associated chunks, summaries, chat messages, quizzes, and activity logs.

---

## 📄 License
This project is licensed under the MIT License.
