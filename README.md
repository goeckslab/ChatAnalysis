# ChatAnalysis: Galaxy Data Analysis Agent

## Introduction

**ChatAnalysis** is an interactive Galaxy tool that lets you perform data analysis via **natural language**. Built on Large Language Models (LLMs) and the **DSPy** prompt framework, ChatAnalysis interprets plain-English requests (e.g. “Generate a heatmap of columns A vs B”) and returns:

- **Data processing** (filtering, aggregations, summary statistics)
- **Visualizations** (plots, charts)
- **Code snippets** (Python / Pandas / AutoGluon)
- **Explanations** (how the analysis was done)

All results appear in a split-pane chat interface **inside Galaxy**—no coding required!

---

## v0.3 Highlights (DSPy Agent)

- **DSPy + Few-Shot Learning**  
  Uses curated Q&A examples to improve prompt accuracy.

- **Split-Pane UI**  
  - **Left:** Conversation  
  - **Right:** Details & Preview (code, tables, plots)  
  Click past queries to revisit full outputs.

- **Faster, Multi-Step Analyses**  
  Optimized logic for chaining data cleaning, visualization, modeling in one session.

- **Advanced ML Support**  
  Integrates AutoGluon, scikit-learn, and more for model training, evaluation, and explanation.

- **Bookmarking & History**  
  Easily revisit previous queries without rerunning; suggested follow-up shortcuts.

---

## Use on Galaxy Main (usegalaxy.org)

1. **Upload Data**  
   Log in to [usegalaxy.org](https://usegalaxy.org), upload your CSV/TSV/Excel file into your history.

2. **Set Your LLM API Key**  
   - Obtain an OpenAI API key (or other supported backend).  
   - In Galaxy: **User → Preferences → OpenAI API Key** → paste your key.

3. **Launch ChatAnalysis**  
   - Find **ChatAnalysis** under **Interactive Tools**.  
   - Select your dataset (and model if prompted) → **Launch**.

4. **Chat & Analyze**  
   - Type queries like “Summarize this dataset.”  
   - View text, code, tables, and plots in the split-pane UI.

5. **Follow-Ups & Refinement**  
   - Context is retained: ask “Filter to 2022 and rerun.”  
   - Click suggested follow-up buttons for one-click analyses.

6. **Stop & Save**  
   - Click **Stop** to end the session.  
   - Outputs are saved back to your Galaxy history.

---

## Install on Your Own Galaxy

### 1. ToolShed Installation (Admin)

- **Repo:** `goeckslab/chatanalysis`  
- Install via ToolShed; enable Interactive Tool support and Docker.

### 2. Local Development

```bash
git clone https://github.com/goeckslab/ChatAnalysis.git
cd ChatAnalysis

# 1. Create & activate a virtual environment
python3.11 -m venv .env
source .env/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r dspy_agent/requirements_nicegui_dspy.txt

# 3. Provide your OpenAI API key
# Option A: Key file (default)
echo YOUR_OPENAI_KEY > dspy_agent/user_config_openai.key

# Option B: CLI argument
# python dspy_agent/chat_dspy.py --openai_key_file /full/path/to/user_config_openai.key

# Option C: provide on the app UI.

# 4. Run the DSPy agent
python dspy_agent/chat_dspy.py

# 5. Open in browser:
# http://localhost:9090
```

_or_

```bash
docker build -t chatanalysis:dspy -f dspy_agent/Dockerfile .
docker run -p 9090:9090 chatanalysis:dspy
# then visit http://localhost:9090
```

> **Note:** Internet access is required for the LLM API.

---

## Legacy Agents (Reference Only)

These prior versions are now **deprecated**—moved to `legacy_agents/`:

- **v0.1** – `pandasai_agent/`  
- **v0.2** – `smolagents_agent/`  

---

## Repository Structure

```
/
├── .github/workflows/        # CI/CD pipelines
├── dspy_agent/               # **v0.3 DSPy agent** (primary code)
├── legacy_agents/            # pandasai_agent/ & smolagents_agent/
├── tools/                    # Galaxy tool XML & support files
├── seq_diagram.md            # Sequence diagram
├── LICENSE
├── README.md
├── .gitignore
└── .dockerignore
```


---

*Feel free to open issues or pull requests for questions or improvements!*
