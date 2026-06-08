# 🏦 Autonomous Customer Profiling & Financial Clustering Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-ReAct_Agent-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Medoids-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**An autonomous AI agent that segments banking customers using advanced unsupervised ML and evaluates new credit applications — powered by the LangChain ReAct architecture.**

[Features](#-features) · [Architecture](#-architecture) · [Installation](#-installation) · [Usage](#-usage) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works)

</div>

---

## 🌟 Overview

This project is a **production-ready, portfolio-grade implementation** of an autonomous financial analysis agent. It combines two powerful paradigms:

1. **LangChain ReAct (Reasoning + Acting)** — the agent reasons about *what* to do, picks the right tool, observes the result, and loops until it has a complete answer. Every Thought, Action, and Observation is fully visible.

2. **Mixed-Data Clustering with Gower Distance + K-Medoids** — because real-world credit datasets contain both numerical (age, credit amount) and categorical (purpose, housing status) features, standard k-means fails. This project uses a statistically rigorous two-stage approach: Gower Distance matrix → K-Medoids (PAM) clustering.

The result: an agent that can autonomously process raw data, discover hidden borrower profiles, and deliver an executive-grade risk opinion on any new applicant.

---

## ✨ Features

### 🤖 Autonomous ReAct Agent
- Built with `langchain` `create_react_agent` + `AgentExecutor`
- Full **Thought → Action → Observation** reasoning trace printed in real-time
- Custom **Senior Financial Analyst persona** baked into the system prompt
- Graceful error handling and self-correction via `handle_parsing_errors`
- Hard iteration cap to prevent runaway loops

### 📊 Advanced Mixed-Data Clustering Pipeline
| Stage | Technique | Why |
|---|---|---|
| Missing value imputation | `"Unknown"` category + median for numerics | Preserves signal in 'Saving accounts' (18.3% NaN) and 'Checking account' (39.4% NaN) |
| Categorical encoding | Ordinal (savings/checking) + One-Hot (nominal) | Respects natural ordering of account levels |
| Feature scaling | `MinMaxScaler` → [0, 1] | Required for Gower distance calculation |
| Distance metric | **Gower Distance** | Handles mixed types natively (Manhattan for num, Dice for binary) |
| Clustering algorithm | **K-Medoids (PAM)** | Uses real data-points as centroids → interpretable and outlier-robust |

### 🔧 Three Specialised LangChain Tools
| Tool | Responsibility |
|---|---|
| `preprocess_german_credit_data` | Load → Impute → Validate → Save cleaned CSV |
| `train_mixed_data_clustering` | Encode → Scale → Gower Matrix → K-Medoids → Profile clusters |
| `analyze_new_customer` | Load pipeline → Transform new row → Nearest-neighbour cluster assignment → Risk report |

### 💾 Persistent Model Artefacts
```
models/
├── clustering_pipeline.joblib   # Fitted encoders, scaler, K-Medoids model
├── preprocessed_data.csv        # Cleaned training data
├── clustered_customers.csv      # Training data + cluster labels
└── cluster_profiles.json        # Per-cluster statistics (mean age, top purpose, etc.)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│   AgentExecutor.invoke({"input": complex_query})                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ReAct Agent Loop                               │
│                                                                  │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │  Thought │ →  │    Action    │ →  │     Tool Call        │  │
│   │ (GPT-4o) │    │ (tool name)  │    │  (Python function)   │  │
│   └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│        ▲                                          │              │
│        └──────────── Observation ─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
┌──────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│  Tool 1          │ │  Tool 2         │ │  Tool 3            │
│  preprocess_     │ │  train_mixed_   │ │  analyze_new_      │
│  german_credit_  │ │  data_          │ │  customer          │
│  data            │ │  clustering     │ │                    │
│                  │ │                 │ │                    │
│  pandas impute   │ │  Gower matrix   │ │  Load pipeline     │
│  save clean CSV  │ │  K-Medoids PAM  │ │  Gower NN search   │
│                  │ │  save pipeline  │ │  Risk report       │
└──────────────────┘ └─────────────────┘ └────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- An OpenAI API key with GPT-4o-mini access

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/autonomous-credit-agent.git
cd autonomous-credit-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and paste your OPENAI_API_KEY

# 5. Verify the dataset is in place
ls data/german_credit_data.csv
```

---

## 🚀 Usage

```bash
python main.py
```

### Expected terminal output

```
╔══════════════════════════════════════════════════════════════════════╗
║   AUTONOMOUS CUSTOMER PROFILING & FINANCIAL CLUSTERING AGENT        ║
╚══════════════════════════════════════════════════════════════════════╝

  Initialising ReAct agent…

╔══════════════════════════════════════════════════════════════════════╗
║                       SUBMITTING QUERY TO AGENT                     ║
╚══════════════════════════════════════════════════════════════════════╝

  Load the dataset at 'data/german_credit_data.csv' and preprocess it.
  Then, segment the customers using a mixed-data clustering algorithm
  (Gower Distance + K-Medoids) with 4 clusters...

> Entering new AgentExecutor chain...

Thought: I need to start by preprocessing the dataset before I can
         train the clustering model or analyse any new customer.
Action: preprocess_german_credit_data
Action Input: data/german_credit_data.csv

Observation:
╔══════════════════════════════════════════════════════════╗
║         PREPROCESSING REPORT – German Credit Data        ║
╚══════════════════════════════════════════════════════════╝
  Dataset shape       : 1,000 rows × 9 columns

  Missing values BEFORE imputation:
    • Saving accounts       :  183 (18.3%)
    • Checking account      :  394 (39.4%)

  Missing values AFTER imputation:
    • Total remaining  : 0 ✅
  ...

Thought: Preprocessing complete. Now I'll train the clustering model.
Action: train_mixed_data_clustering
Action Input: 4

Observation:
╔══════════════════════════════════════════════════════════╗
║       CLUSTERING TRAINING REPORT – K-Medoids / Gower    ║
╚══════════════════════════════════════════════════════════╝
  Algorithm : K-Medoids (PAM) + Gower Distance
  Clusters  : 4
  Samples   : 1,000

============================================================
  Cluster 0  (n=287)
============================================================
  Avg Age           : 38.4 yrs
  Avg Credit Amount : €2,134
  Avg Loan Duration : 16.2 months
  Top Purpose       : radio/TV
  Risk Assessment   : ✅ LOW RISK – established customers...

  Cluster 1  (n=198)
  ...

Thought: Training complete. Now I'll analyse the new applicant.
Action: analyze_new_customer
Action Input: {"Age": 24, "Sex": "male", "Job": 2, "Housing": "rent",
               "Saving accounts": "little", "Checking account": "moderate",
               "Credit amount": 4500, "Duration": 36, "Purpose": "business"}

Observation:
╔══════════════════════════════════════════════════════════╗
║          NEW CUSTOMER ANALYSIS – CLUSTER ASSIGNMENT      ║
╚══════════════════════════════════════════════════════════╝

  Assigned Cluster ID  : 2
  Cluster Size         : 243 historical customers
  ...
  Risk Assessment   : ⚡ MEDIUM RISK – moderate financial stability
  ...
  CAUTION ADVISED. Applicant is 14 years younger than typical cluster
  member → limited credit history likely. Credit amount €4,500 is
  €1,200 above cluster average → elevated exposure.

Thought: I now have all the information needed for a complete answer.
Final Answer: ...

╔══════════════════════════════════════════════════════════════════════╗
║                       FINAL AGENT RESPONSE                          ║
╚══════════════════════════════════════════════════════════════════════╝

  The 24-year-old male applicant requesting €4,500 over 36 months for
  a business purpose has been assigned to Cluster 2, which represents
  younger borrowers with limited savings and moderate checking balances
  seeking larger business or education loans...

  ✅  Pipeline artefacts written to: models/
  ✅  Run complete.
```

---

## 📁 Project Structure

```
autonomous-credit-agent/
│
├── agents/
│   ├── __init__.py
│   └── react_agent.py          # ReAct agent factory (LLM + tools + prompt)
│
├── tools/
│   ├── __init__.py
│   └── clustering_tools.py     # 3 LangChain @tool functions
│
├── data/
│   └── german_credit_data.csv  # Raw dataset (1,000 customers, 9 features)
│
├── models/                     # Auto-created on first run
│   ├── clustering_pipeline.joblib
│   ├── preprocessed_data.csv
│   ├── clustered_customers.csv
│   └── cluster_profiles.json
│
├── notebooks/                  # (Optional) EDA notebooks
│
├── main.py                     # Entry point
├── requirements.txt
├── .env.example                # Template for API key config
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

### 1. Dataset
The [German Credit Dataset](https://www.kaggle.com/datasets/uciml/german-credit) contains 1,000 historical loan applicants with 9 features:

| Feature | Type | Notes |
|---|---|---|
| Age | Numerical | Customer age in years |
| Sex | Categorical | male / female |
| Job | Ordinal (0–3) | 0 = unskilled, 3 = highly skilled |
| Housing | Categorical | own / rent / free |
| Saving accounts | Ordinal | little / moderate / quite rich / rich |
| Checking account | Ordinal | little / moderate / rich |
| Credit amount | Numerical | Loan amount in Deutsche Marks |
| Duration | Numerical | Loan duration in months |
| Purpose | Categorical | car / radio/TV / education / business / … |

### 2. Why Gower Distance?
Standard distance metrics (Euclidean, Manhattan) cannot meaningfully combine numerical and categorical variables. **Gower Distance** solves this:
- For **numerical** columns: normalised Manhattan distance
- For **categorical/binary** columns: Dice similarity (0 if same, 1 if different)
- The final distance is a weighted average across all features → values in [0, 1]

### 3. Why K-Medoids over K-Means?
| Property | K-Means | K-Medoids (PAM) |
|---|---|---|
| Centroid type | Abstract mean vector | Actual data point |
| Outlier robustness | Low | High |
| Mixed data support | ❌ (needs precomputed matrix) | ✅ (`metric="precomputed"`) |
| Interpretability | Low | High (medoid = representative customer) |

### 4. ReAct Loop
The agent uses **chain-of-thought prompting** inside the ReAct loop:
```
Thought  → "I need to preprocess first, then train, then infer"
Action   → preprocess_german_credit_data("data/german_credit_data.csv")
Obs      → "1,000 rows cleaned, 0 nulls remaining…"
Thought  → "Good, now train clustering…"
Action   → train_mixed_data_clustering(4)
Obs      → "4 clusters discovered…"
Thought  → "Now analyse the new applicant…"
Action   → analyze_new_customer({...})
Obs      → "Assigned to Cluster 2, MEDIUM RISK…"
Thought  → "I now have a complete answer."
Final    → Executive risk report
```

---

## 🔬 Extending the Project

| Idea | How |
|---|---|
| Add a `visualize_clusters` tool | Use UMAP → 2D plot saved as PNG |
| Swap to K-Prototypes | Replace Gower+KMedoids with `kmodes.KPrototypes` |
| Add LangSmith tracing | Set `LANGCHAIN_TRACING_V2=true` in `.env` |
| Use GPT-4o for deeper reasoning | Change `model_name="gpt-4o"` in `main.py` |
| REST API wrapper | Wrap `main.py` logic with FastAPI |
| CI/CD | Add GitHub Actions with `pytest` for tool unit tests |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [German Credit Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [LangChain ReAct Agent documentation](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [Gower distance paper – Gower (1971)](https://www.jstor.org/stable/2528823)
- [K-Medoids (PAM) – scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html)

---

<div align="center">
  <i>Built as a portfolio-grade demonstration of autonomous AI agents + advanced unsupervised learning for financial risk analysis.</i>
</div>
