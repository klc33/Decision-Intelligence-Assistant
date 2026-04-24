# Decision Intelligence Assistant

A full‑stack AI assistant that compares **Retrieval‑Augmented Generation (RAG)**, a **Machine Learning baseline**, and **LLM zero‑shot predictions** on real customer support data from Twitter.

The system answers user questions by retrieving similar past support tickets, generates answers with and without context, predicts ticket priority using both a trained classifier and an LLM, and presents a four‑way comparison of accuracy, latency, and cost.

---

## Architecture

┌──────────────┐ ┌──────────────────────────────────────┐
│ Frontend │────▶│ Backend (FastAPI) │
│ (React) │ │ │
│ :3000 │ │ ┌──────────┐ ┌──────────┐ │
└──────────────┘ │ │ RAG │ │ Non‑RAG │ │
│ │ Service │ │ Service │ │
│ └─────┬─────┘ └─────┬─────┘ │
│ │ │ │
│ ┌─────▼─────┐ ┌────▼──────┐ │
│ │ ChromaDB │ │ Groq LLM │ │
│ │ (Vector │ │ (llama │ │
│ │ Store) │ │ 3.1-8B) │ │
│ └───────────┘ └───────────┘ │
│ │
│ ┌──────────┐ ┌──────────┐ │
│ │ ML Model │ │ LLM │ │
│ │ Priority │ │ Priority │ │
│ │ (sklearn)│ │ (zero‑ │ │
│ │ │ │ shot) │ │
│ └──────────┘ └──────────┘ │
└──────────────────────────────────────┘



### Data Flow
1. User types a query in the React frontend.
2. Backend retrieves **top‑k similar past tickets** from ChromaDB.
3. **RAG Answer:** LLM generates a response using retrieved context.
4. **Non‑RAG Answer:** LLM generates a response without context.
5. **ML Priority:** Trained classifier predicts urgent/normal with confidence.
6. **LLM Priority:** Zero‑shot LLM call predicts urgent/normal.
7. All four outputs are returned with latency and cost metrics.
8. Frontend displays answers, sources, and a comparison panel.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Axios |
| **Backend** | FastAPI, Uvicorn |
| **Vector Store** | ChromaDB (persistent mode) |
| **Embeddings** | Sentence‑Transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Groq API (`llama-3.1-8b-instant`) |
| **ML Model** | scikit‑learn (Logistic Regression / Random Forest / XGBoost) |
| **Containerization** | Docker, Docker Compose |
| **Package Manager** | `uv` (pip compatible) |

---

## Project Structure

decision-intelligence-assistant/
├── README.md
├── .env.example
├── docker-compose.yml
├── .gitignore
├── notebooks/ # EDA, labeling experiments, RAG tests
├── training/ # ML model training scripts
│ ├── labeling.py # Weak supervision labeling function
│ ├── features.py # Feature engineering
│ ├── train.py # Train/val/test split + GridSearchCV
│ └── outputs/ # Saved model and metrics (generated)
├── scripts/ # Data preparation and vector store build
│ ├── prepare_data.py # Clean + label 500k tweets
│ └── build_solutions_store.py # Populate ChromaDB with company replies
├── data/ # Raw and processed data (generated)
├── chroma_data/ # Persistent vector store (generated)
├── backend/
│ ├── Dockerfile
│ ├── requirements.txt
│ └── app/
│ ├── main.py
│ ├── config.py
│ ├── schemas.py
│ ├── routers/
│ ├── services/
│ └── utils/
└── frontend/
├── Dockerfile
├── nginx.conf
└── src/
├── App.js
├── App.css
└── components/



---

## Dependencies

### Backend (Python)
- FastAPI, Uvicorn
- ChromaDB
- Sentence‑Transformers
- Groq
- scikit‑learn, XGBoost
- Pandas, NumPy, Joblib
- Pydantic, python‑dotenv
- TextBlob (sentiment features)

### Frontend (Node)
- React 18
- Axios (HTTP client)

---

## Setup & Running Locally

### Prerequisites
- Python 3.13+
- Node.js 18+
- Groq API key ([console.groq.com](https://console.groq.com))

### 1. Clone and set up environment
```bash
git clone https://github.com/klc33/Decision-Intelligence-Assistant
cd decision-intelligence-assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r backend/requirements.txt

2.
cp .env.example .env
# Edit .env and add your GROQ_API_KEY



# Download twcs.csv from Kaggle and place in data/raw/
# https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
3.
python scripts/prepare_data.py          # ~500k cleaned customer tweets
python training/train.py                # Train ML classifier
python scripts/build_solutions_store.py # Build ChromaDB with company replies

4.
cd backend
uvicorn app.main:app --reload --port 8000

5.
cd frontend
npm install
npm start



6. Open the app
Frontend: http://localhost:3000

API Docs: http://localhost:8000/docs



Running with Docker
Prerequisites
Docker Desktop installed

Steps 1‑3 from "Setup & Running Locally" completed (data, model, vector store generated)

Build and start


docker compose up --build




#to view logs

docker compose logs backend -f
docker compose logs frontend -f