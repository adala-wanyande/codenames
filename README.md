# Evaluating LLM-Based Agents in Codenames

This repository contains the code for our **Modern Game AI Algorithms** course project (Leiden University). We built a fully automated evaluation framework to simulate the board game Codenames, systematically comparing traditional embedding-based methods (Word2Vec/GloVe) against open-weight Large Language Models (Qwen, LLaMA-3, Mistral) operating under varying levels of prompt complexity.

## Team Members
* **Irshad Bakhtali:** API \& Model Integration
* **Irene Chrysovergi:** Prompt Engineering \& Simulation Search
* **Reshit Fazlija:** Evaluation Framework \& Ablations
* **Benard Wanyande:** Core Python Engine \& Baseline Architectures
* **Lucas Zuurmond:** Data Logging \& Visualizations

---

## Local Setup Guide

Follow these steps to configure your Python environment, safely store API keys, and download the necessary local AI models.

### Step 1: Clone the Repository
```bash
git clone https://github.com/adala-wanyande/codenames.git
cd codenames
```

### Step 2: Set Up the Virtual Environment
We use a virtual environment to ensure dependency consistency.

**For Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**For Windows (Command Prompt / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

Once activated, install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: API Keys (Optional for Cloud Models)
If you wish to test cloud models (Gemini/Claude), you must provide API keys safely via a `.env` file. **Never upload your actual API keys to GitHub.**
1. Duplicate the `.env.example` file and rename it to `.env`.
2. Paste your keys inside:
   `GEMINI_API_KEY=your_actual_gemini_key_here`

### Step 4: Setting Up Local Models (Ollama)
Our primary experiments run locally to ensure zero-cost, reproducible tournament scaling. 
1. Install [Ollama](https://ollama.com/).
2. Open your terminal and download the baseline evaluation model:
```bash
ollama run qwen2.5
```
*(Type `/bye` to exit once it finishes downloading).*

---

## Running the Code

Our framework supports running individual sandbox matches as well as large-scale, automated benchmarking tournaments.

### Option A: Run a Single Sandbox Match
You can run a single game using `src.main`. This is useful for debugging specific prompt depths or shot counts. The script accepts two arguments: `--spymaster_type` and `--shot`.

**Single Chain-of-Thought (Zero or Few-Shot):**
```bash
python3 -m src.main --spymaster_type single_cot --shot 1
```
**Double Chain-of-Thought (Separates Strategic Grouping and Clue Generation):**
```bash
python3 -m src.main --spymaster_type double_cot --shot 1
```
**Stochastic Rollout (MCTS-inspired Lookahead):**
```bash
python3 -m src.main --spymaster_type double_cot_SR --shot 1
```

### Option B: Run the Full Tournament
To replicate the results from our academic paper, you can run the full tournament suite. This script loops through all agent configurations, plays `n_games` for each on fixed benchmark boards, logs the metrics to a CSV, and prints a final summary.

```bash
python3 -m src.tournament
```
*(Note: You can adjust `n_games` directly inside `src/tournament.py` before running).*

### Option C: Generate Visualizations
Once the tournament finishes and `data/logs/tournament_results.csv` is populated, you can generate the academic charts used in our report:
```bash
python3 -m evaluate.visualize
```
Plots will be saved directly to the `data/figures/` directory.

---

## 📂 Project Structure

```text
codenames/
├── src/
│   ├── engine/                 # Game logic, board state, strict rules (Numpy)
│   ├── agents/                 # LLM and Word2Vec agent architectures
│   ├── utils/                  # Data logging, metrics, config factory
│   ├── main.py                 # Single-game execution CLI
│   └── tournament.py           # Automated multi-agent tournament runner
├── evaluate/
│   └── visualize.py            # Seaborn/Matplotlib visualization generator
├── data/
│   ├── logs/                   # Raw CSV outputs of tournament metrics
│   └── figures/                # Generated academic plots
├── tests/                      # Pytest suite enforcing game rules
├── docs/                       # LNCS LaTeX report
├── .env.example                # Template for environment variables
├── .gitignore                  # Keeps our API keys and data safe
└── requirements.txt            # Python dependencies
```

## Git Workflow
To avoid merge conflicts and preserve the integrity of the game engine, **please do not push directly to the `main` branch.** 

1. Ensure you are up to date: `git checkout main` $\rightarrow$ `git pull`
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Write your code, test it (`pytest`), and commit.
4. Push your branch: `git push origin feature/your-feature-name`
5. Open a **Pull Request (PR)** on GitHub and request a review before merging.
