# CTRL-INTEL: Backend API & Orchestration Engine

**Team GenIQ** | **Schneider University Hackathon**

This repository contains the high-performance **FastAPI** backend for CTRL-INTEL. It serves as the central nervous system for the platform, managing file systems, database access, and orchestrating the autonomous Multi-Agent AI pipeline.

## System Overview

The backend implements a sequential data refinement pipeline where raw Control Narratives are transformed into validated IEC 61131-3 Structured Text. It utilizes **CrewAI** and **LangChain**, powered by Google Gemini models, to coordinate a team of specialized agents.

The system relies on a **Dual-Database** strategy:

* **MongoDB:** Stores user data, file paths, and session management.
* **ChromaDB:** A Vector Database used to embed and store parsed markdown files for efficient retrieval.

## Project Structure & Agent Mapping

The codebase is organized to reflect the "Factory Model" architecture described in the documentation.

```text
📂 control_narrative_backend
├── 📄 main.py                      # Application entry point (FastAPI)
├── 📄 config.py                    # Environment configuration
├── 📁 app
│   ├── 📁 agents                   # THE AGENTIC CORE [cite: 34]
│   │   ├── 📄 pipeline.py          # The Orchestration Layer (Plant Manager) [cite: 396]
│   │   ├── 📄 document_agent.py    # Ingestion: Manages PDF parsing strategies [cite: 30]
│   │   ├── 📁 parsers              # Dual-Engine Architecture [cite: 161]
│   │   │   ├── 📄 deterministic_parser.py # Strategy A: Fast rule-based heuristic [cite: 164]
│   │   │   └── 📄 docling_parser.py       # Strategy B: AI-powered layout analysis [cite: 173]
│   │   ├── 📄 image_parser_agent.py  # Vision: Qwen-VL based diagram analysis [cite: 32, 201]
│   │   ├── 📄 understanding_agent.py # Processing: Map-Reduce system summarization [cite: 35]
│   │   ├── 📄 control_logic_agent.py # Extraction: Identifies rules and tags [cite: 36]
│   │   ├── 📄 mapper_agent.py      # Strategy: Loop Mapper / Topology Architect [cite: 37, 297]
│   │   ├── 📄 validator_agent.py   # QA: Data Fusion & Safety Verification [cite: 38, 327]
│   │   └── 📄 code_generator_agent.py # Output: IEC 61131-3 ST generation [cite: 39, 368]
│   ├── 📁 chat                     # INTERACTIVE SUPPORT
│   │   ├── 📄 rag_engine.py        # Implements RAG, ReACT, and FLARE strategies [cite: 47]
│   │   └── 📄 routes.py            # Chat API endpoints
│   ├── 📁 documents                # FILE MANAGEMENT
│   │   ├── 📄 parsed_crud.py       # Handling intermediate JSON/Markdown outputs
│   │   └── 📁 user_documents       # Local storage for raw uploads [cite: 43]
│   ├── 📁 auth                     # SECURITY
│   │   ├── 📄 jwt_handler.py       # Token management
│   │   └── 📄 routes.py            # Login/Register endpoints
│   └── 📁 utils
│       ├── 📄 embeddings.py        # Vector embedding logic (ChromaDB)
│       └── 📄 io_manager.py        # File system operations
└── 📁 db                           # DATABASE CONNECTION
    └── 📄 database.py              # MongoDB connection logic

```

## Key Modules

### 1. The Orchestration Layer (`app/agents/pipeline.py`)

This script acts as the "Plant Manager". It enforces the sequential dependency model where the output of one agent becomes the mandatory input for the next. It manages the "Baton Pass" from Ingestion → Understanding → Logic → Mapping → Validation → Code Generation.

### 2. Dual-Engine Parser (`app/agents/parsers/`)

The `document_agent.py` dynamically selects between:

* **Deterministic Parser:** Uses `PyMuPDF` for speed on standard digital PDFs.
* **Docling Parser:** Uses IBM Docling for high-fidelity extraction of complex/scanned layouts.

### 3. Vision Analysis (`image_parser_agent.py`)

Uses the **Qwen Gemma 3VL Model** via Ollama to "see" P&IDs and electrical diagrams. It performs context-aware analysis to translate visual assets into descriptive text.

### 4. Code Generation (`code_generator_agent.py`)

Implements a "Software Factory Model". It writes the entire program file in a "Single-Shot Synthesis," ensuring global variable coherence and generating compliant Structured Text.

## Setup & Installation

### Prerequisites

* Python 3.10+
* MongoDB (Running locally or cloud URI)
* Ollama (For local Qwen/Gemma models)
* Google Gemini API Key

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-repo/ctrl-intel-backend.git
cd ctrl-intel-backend

```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. **Install Dependencies**
```bash
pip install -r requirements.txt

```

4. **Environment Configuration**
Create a `.env` file in the root directory:
```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=control_narrative_db
GEMINI_API_KEY=your_google_api_key
SECRET_KEY=your_jwt_secret
ALGORITHM=HS256
CHROMA_DB_PATH=./chroma_db

```

5. **Run the Server**
```bash
uvicorn main:app --reload

```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, access the interactive Swagger UI at:
**`http://localhost:8000/docs`**

### Core Endpoints:

* `POST /auth/login`: Authenticate and receive JWT.
* `POST /documents/upload`: Upload raw PDF.
* `GET /documents/{id}/status`: Check pipeline progress (uses `update_progress` hooks).
* `POST /chat/query`: Interactive query using RAG/ReACT.

## Future Improvements

* **Vendor-Specific Export:** Extend `code_generator_agent.py` to support `.L5X` (Rockwell) and `.AWL` (Siemens) formats.
* **Ladder Logic Generation:** Fine-tune models to generate XML-based Ladder Logic alongside ST.
* **Self-Learning:** Implement feedback loops in `validator_agent.py` to store user corrections in ChromaDB.
