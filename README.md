
# Evaluating LLM-Based Agents in Codenames

This repository contains the code for our **Modern Game AI Algorithms** course project (Leiden University). We are building and evaluating AI agents to play the board game Codenames, systematically comparing traditional embedding-based methods (Word2Vec/GloVe) against massive cloud LLMs (Gemini, Claude) and efficient local LLMs (Qwen).

## Team Members
Currently, we are all collaborating across the codebase to build the core game engine, set up API connections, and design the initial agents:
* Irshad Bakhtali
* Irene Chrysovergi
* Reshit Fazlija
* Benard Wanyande
* Lucas Zuurmond

---

## Getting Started: Local Setup Guide

Follow these steps exactly to get the project running on your local machine. This setup requires configuring a Python environment, safely storing API keys, and downloading local AI models.

### Step 1: Clone the Repository
Open your terminal (or VS Code terminal) and run:
```bash
git clone https://github.com/adala-wanyande/codenames.git
cd codenames
```

### Step 2: Set Up the Virtual Environment
We use a virtual environment to ensure everyone is using the exact same package versions and to avoid cluttering your computer's global Python installation.

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
*You will know it worked if you see `(.venv)` at the beginning of your terminal line.*

Once activated, install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

### Step 3: Setting Up API Keys (Gemini & Claude)
We use Google Gemini and Anthropic Claude for our cloud LLMs. **Crucial Rule: NEVER upload your actual API keys to GitHub.** 

We use a `.env` file to keep them safe on your local machine only.
1. Look in the main folder for a file named `.env.example`.
2. Duplicate this file and rename the new copy to strictly `.env`.
3. Open `.env` and paste your API keys inside so it looks like this:
```text
GEMINI_API_KEY=your_actual_gemini_key_here
ANTHROPIC_API_KEY=your_actual_claude_key_here
```
*(Note: If you don't have personal API keys yet, you can leave these blank for now and just use the local Ollama models in Step 4).* Our `.gitignore` file is already configured to hide your `.env` file from Git.

---

### Step 4: Setting Up Local Models (Ollama)
To run massive AI tournaments for free and ensure experimental reproducibility, we are running open-weight models locally on our own machines using Ollama.

1. **Download Ollama:** Go to [ollama.com](https://ollama.com/) and download the app for your operating system.
2. **Install and Run:** Install the app and make sure the Ollama application is actually open/running in the background of your computer (you should see its icon in your system tray/menu bar).
3. **Download the Qwen 2.5 Model:** Open your terminal and run the following command. This will download the model weights (a few gigabytes, so it may take a few minutes):
```bash
ollama run qwen2.5
```
4. **Test it:** Once it finishes downloading, it will open a chat prompt in your terminal. Say "Hello". Once it replies, type `/bye` to exit. 
*Ollama is now permanently installed on your machine and Python can talk to it!*

---

### Step 5: The Sanity Check
To ensure your Python environment, API keys, and local Ollama models are set up perfectly, run the test script from the main folder:

**Mac/Linux:** `python3 -m src.main`  
**Windows:** `python -m src.main`  

If everything is working, you should see the Codenames board generate in the terminal, along with dummy outputs from the Agent files!

---

## Project Structure

```text
codenames/
├── src/                        # Main Python source code
│   ├── engine/                 # Game logic, board state, rules (Numpy)
│   ├── agents/                 # Agent logic (LLMs and Baselines)
│   ├── utils/                  # Data logging, metrics (Pandas)
│   └── main.py                 # Tournament execution & entry point
├── data/                       # Local data (Ignored by Git)
├── notebooks/                  # Jupyter notebooks for data visualization
├── docs/                       # Final LNCS LaTeX report files
├── .env.example                # Template for environment variables
├── .gitignore                  # Keeps our API keys and data safe
└── requirements.txt            # Python dependencies
```

## Git Workflow (Important!)
To avoid merge conflicts and breaking the code, **please do not push directly to the `main` branch.** 

When starting a new feature or task, follow this workflow:
1. Get the latest code: `git checkout main` and then `git pull`
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Write your code, test it, and commit.
4. Push your branch: `git push origin feature/your-feature-name`
5. Go to GitHub and open a **Pull Request (PR)** so the team can review it before merging.
```