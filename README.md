# 🤖 Multi Agent QA for Financial Documents

<div align="center">

**An intelligent LangGraph-based multi-agent system for financial document analysis, powered by LLM orchestration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Agent Descriptions](#-agent-descriptions)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements a sophisticated multi-agent system for financial document analysis using **LangGraph** orchestration. The system intelligently routes queries through specialized agents, each responsible for distinct tasks like document retrieval, web search, data extraction, calculations, and answer generation.

Built for the **Inter IIT Tech Meet**, this system demonstrates advanced AI agent coordination, retrieval-augmented generation (RAG), and dynamic query routing.

### What Makes This Special?

- **🧠 Intelligent Intent Classification**: Distinguishes between casual queries, conversation history requests, and informational questions
- **🔄 Dynamic Agent Orchestration**: Agents autonomously decide next steps based on query requirements
- **📊 Multi-Modal Processing**: Handles document retrieval, web search, table extraction, and mathematical calculations
- **💾 Thread-Aware Caching**: Maintains conversation context across sessions
- **⚡ Fault-Tolerant Design**: Graceful error handling with automatic fallback mechanisms

---

## ✨ Key Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Agent System** | 8 specialized agents working in coordinated swarm architecture |
| **RAG Pipeline** | Retrieval-augmented generation with vector store and web search |
| **Intent Classification** | LLM-based query classification (casual/informational/history) |
| **Smart Routing** | Dynamic agent handoffs based on query type and data availability |
| **Financial Analysis** | Specialized in extracting numbers, calculating ratios, and comparisons |
| **Conversation Memory** | Thread-aware caching and conversation history tracking |
| **PDF Processing** | Advanced document parsing and chunking |
| **Web Search Integration** | Real-time external data retrieval via Tavily API |

### 🛠️ Technical Features

- **Stateful Graph Execution**: LangGraph StateGraph with persistent memory
- **Retry Logic**: Exponential backoff for transient failures
- **Error Handling**: Comprehensive exception handling with fallback agents
- **Type Safety**: Pydantic models for state validation
- **Modular Design**: Clean separation of concerns across agents
- **Streamlit Interface**: User-friendly web UI with file upload

---

## 🏗️ System Architecture

### Agent Flow Diagram

```
User Query
    ↓
┌─────────────┐
│   Memory    │ ← Checks cache for similar queries
└──────┬──────┘
       ↓
┌─────────────┐
│  Retriever  │ ← Searches vector store for documents
└──────┬──────┘
       ↓
┌─────────────┐
│  Validator  │ ← Classifies intent & validates data
└──────┬──────┘
       ↓
    ┌──┴──┐
    │     │
    ↓     ↓
┌────────┐  ┌──────────┐
│WebSearch│  │Summarizer│
└───┬────┘  └────┬─────┘
    │            │
    ↓            ↓
┌─────────┐  ┌────────┐  ┌──────┐
│  Table  │→ │  Math  │→ │Aggr. │ → Final Answer
└─────────┘  └────────┘  └──────┘
```

### Agent Responsibilities

```
┌──────────────────────────────────────────────────────────────┐
│ Memory Agent                                                  │
│ -  Query cache lookup using vector similarity                 │
│ -  Thread-aware conversation history                          │
├──────────────────────────────────────────────────────────────┤
│ Retriever Agent                                              │
│ -  Document retrieval from Chroma vector store                │
│ -  Semantic search with embeddings                            │
├──────────────────────────────────────────────────────────────┤
│ Validator Agent                                              │
│ -  Intent classification (casual/informational/history)       │
│ -  Query type detection (calculation/summary/general)         │
│ -  Data relevance validation                                  │
├──────────────────────────────────────────────────────────────┤
│ WebSearch Agent                                              │
│ -  External data retrieval via Tavily API                     │
│ -  Fallback for insufficient document data                    │
├──────────────────────────────────────────────────────────────┤
│ Summarizer Agent                                             │
│ -  Concise content summarization                              │
│ -  Handles large document contexts                            │
├──────────────────────────────────────────────────────────────┤
│ Table Agent                                                  │
│ -  Structured numeric data extraction                         │
│ -  JSON formatting of financial metrics                       │
├──────────────────────────────────────────────────────────────┤
│ Math Agent                                                   │
│ -  Financial calculations (ratios, growth rates, etc.)        │
│ -  Comparative analysis                                       │
├──────────────────────────────────────────────────────────────┤
│ Aggregator Agent                                             │
│ -  Final answer synthesis                                     │
│ -  Error handling and user-friendly responses                 │
│ -  Query caching for future lookups                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- API keys (GROQ, Tavily)

### Step 1: Clone the Repository

```
git clone https://github.com/Aditya-ad48/INTERIIT-NLP-Prepathon.git
cd INTERIIT-NLP-Prepathon
```

### Step 2: Create Virtual Environment

```
# Create virtual environment
python -m venv finance_env

# Activate (Mac/Linux)
source finance_env/bin/activate

# Activate (Windows)
finance_env\Scripts\activate
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get API Keys:**
- **GROQ**: [console.groq.com](https://console.groq.com)
- **Tavily**: [app.tavily.com](https://app.tavily.com)

### Step 5: Prepare Document Store

```
# Create directory for PDFs
mkdir -p data

# Add your financial documents (PDFs) to the data/ folder
# Example: data/financial_report_q1.pdf

# Ingest documents into vector store
python ingest.py
```

---

## 💻 Usage

### Running the Application

```
streamlit run app_streamlit.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Upload Documents** (optional)
   - Click "Browse files" to upload PDF documents
   - Documents are processed and added to the vector store

2. **Ask Questions**
   - Type your query in the chat input
   - Examples:
     - "What was the Q1 revenue?"
     - "Calculate the profit margin for 2024"
     - "Compare expenses between Q1 and Q2"
     - "Summarize the risk factors"

3. **View Results**
   - See the agent execution trace
   - Get comprehensive answers with sources
   - Cached responses load instantly for repeated queries

### Example Queries

**Financial Analysis:**
```
"What is the total revenue for Q1 2024?"
"Calculate the year-over-year growth rate"
"Compare operating expenses between quarters"
```

**Document Summarization:**
```
"Summarize the key risks mentioned in the report"
"Give me an overview of the financial performance"
```

**Casual Interaction:**
```
"Hello"
"Thank you"
"What did I ask earlier?"
```

---

## 🤖 Agent Descriptions

### 1. Memory Agent
**Purpose**: Query cache and conversation history management

- Searches cache using vector similarity
- Returns cached answers for duplicate queries
- Maintains thread-aware conversation history
- Reduces latency and API costs

**Handoffs**: → Retriever (on cache miss)

---

### 2. Retriever Agent
**Purpose**: Document retrieval from vector store

- Semantic search using HuggingFace embeddings
- Retrieves top-k relevant document chunks
- Validates document quality
- Fallback to WebSearch on failure

**Handoffs**: → Validator (always)

---

### 3. Validator Agent
**Purpose**: Intent classification and routing orchestration

**Three-tier classification:**
1. **Intent Classification**: Casual vs Informational vs History
2. **Query Type Detection**: Calculation vs Summary vs General
3. **Data Relevance Validation**: Ensures data can answer query

**Handoffs**: → WebSearch | Summarizer | Table | Aggregator

---

### 4. WebSearch Agent
**Purpose**: External data retrieval

- Tavily API integration for web search
- Fallback when documents are insufficient
- Returns relevant web content
- Handles API errors gracefully

**Handoffs**: → Validator (for assessment)

---

### 5. Summarizer Agent
**Purpose**: Content summarization

- Generates concise summaries
- Handles large contexts (30K+ chars)
- Extracts key points
- Uses LLM for natural language generation

**Handoffs**: → Aggregator (always)

---

### 6. Table Agent
**Purpose**: Structured data extraction

- Extracts numeric data (revenue, expenses, ratios)
- Formats as JSON with metadata
- Preserves temporal context (periods)
- Identifies calculation requirements

**Handoffs**: → Math | Aggregator

---

### 7. Math Agent
**Purpose**: Financial calculations

**Capabilities:**
- Comparisons (Q1 vs Q2)
- Growth rates & percentage changes
- Financial ratios (profit margin, etc.)
- Aggregations (totals, averages)
- Step-by-step calculation breakdown

**Handoffs**: → Aggregator (always)

---

### 8. Aggregator Agent
**Purpose**: Final answer synthesis

- Combines information from all sources
- Generates natural language responses
- Handles error cases gracefully
- Caches answers for future queries
- Responds to casual queries directly

**Handoffs**: → END (terminal node)

---

## 📁 Project Structure

```
INTERIIT-NLP-Prepathon/
│
├── agents/                      # Agent implementations
│   ├── __init__.py
│   ├── memory_agent.py         # Cache & conversation history
│   ├── retriever_agent.py      # Document retrieval
│   ├── validator_agent.py      # Intent classification & routing
│   ├── websearch_agent.py      # External data retrieval
│   ├── summarizer_agent.py     # Content summarization
│   ├── table_agent.py          # Data extraction
│   ├── math_agent.py           # Financial calculations
│   └── aggregator_agent.py     # Answer synthesis
│
├── config.py                    # Configuration & environment variables
├── models.py                    # Pydantic models & type definitions
├── decorators.py                # Error handling & retry logic
├── utils.py                     # Utility functions
├── pdf_utils.py                 # PDF processing utilities
│
├── graph.py                     # LangGraph orchestration
├── ingest.py                    # Document ingestion pipeline
├── app_streamlit.py             # Streamlit web interface
│
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | GROQ LLM API key | ✅ Yes |
| `TAVILY_API_KEY` | Tavily search API key | ✅ Yes |

### Configurable Parameters

Edit `config.py` to customize:

```
# LLM Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector Store
DEFAULT_DB_PATH = "vectorstore_final"

# Retrieval Settings
RETRIEVAL_K = 10  # Number of documents to retrieve

# Timeouts
AGENT_TIMEOUT = 30  # seconds
```

---

## 🛠️ Technologies Used

### Core Frameworks
- **LangGraph**: Multi-agent orchestration and state management
- **LangChain**: LLM abstraction and RAG pipeline
- **Streamlit**: Web interface

### AI & ML
- **GROQ**: High-performance LLM inference
- **Chroma**: Vector database for semantic search
- **HuggingFace**: Embedding models

### APIs
- **Tavily**: Web search API
- **Unstructured**: PDF parsing and extraction

### Data Processing
- **Pandas**: Data manipulation
- **PyPDF**: PDF document handling

---

## 🎓 Learning Resources

This project demonstrates:

- **LangGraph State Management**: Persistent conversation state
- **Multi-Agent Coordination**: Autonomous agent handoffs
- **RAG Architecture**: Retrieval-augmented generation
- **Intent Classification**: LLM-based query understanding
- **Error Handling**: Production-grade fault tolerance
- **Caching Strategies**: Vector similarity-based cache

---

## 📊 Performance

- **Average Response Time**: 2-5 seconds (first query)
- **Cache Hit Response**: < 0.5 seconds
- **Document Retrieval**: ~1 second for 10 documents
- **Web Search Fallback**: 2-3 seconds

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Aditya Ahirwar**
- GitHub: [@Aditya-ad48](https://github.com/Aditya-ad48)
- Email: adityaraj13.ahirwar@gmail.com

---

## 🙏 Acknowledgments

- **Inter IIT Tech Meet** for the opportunity
- **LangChain Team** for LangGraph framework
- **GROQ** for high-performance LLM inference
- **Tavily** for web search capabilities

---

<div align="center">

**Built with ❤️ for Inter IIT Tech Meet 2025**

⭐ Star this repo if you found it helpful!

</div>

---

## 📚 Research Papers & Foundations

This project implements concepts from the following research papers:

### Core Architecture

1. **Retrieval-Augmented Generation (RAG)**
   - Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - Meta AI Research
   - [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
   - **Implementation**: Document retrieval with Chroma vector store + LLM generation

2. **LangGraph Multi-Agent Orchestration**
   - Duan & Wang (2024). "Exploration of LLM Multi-Agent Application Implementation Based on LangGraph+CrewAI"
   - [arXiv:2411.18241](https://arxiv.org/html/2411.18241v1)
   - **Implementation**: StateGraph with 8 specialized agents and conditional routing

### Supporting Concepts

3. **Swarm Intelligence for Coordination**
   - Gnanamani & Kumaravel (2025). "Coordination and Collaboration in Multi-Agent Autonomous Systems: A Swarm Intelligence Approach"
   - **Implementation**: Decentralized agent decision-making and autonomous handoffs

4. **Advanced RAG Techniques**
   - Gao et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey"
   - [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
   - **Implementation**: Thread-aware caching, semantic similarity search

### Technical Innovations

- **Intent Classification**: LLM-based query routing
- **Task Specialization**: Domain-specific agents (financial analysis, calculations)
- **Fault Tolerance**: Exponential backoff retry logic
- **State Persistence**: LangGraph checkpointing with MemorySaver
