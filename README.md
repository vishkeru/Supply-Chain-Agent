# Supply Chain Diagnostic Agent

**An AI analyst that answers supply chain questions like a consultant: it builds an issue tree, tests every branch with data, sizes the real drivers and recommends prioritised actions, and it never invents a number.**

 ## Demo video

https://github.com/user-attachments/assets/7796ae54-b779-4564-936d-c9f49abfd3a8

## Executive summary from a saved run
<img width="919" height="425" alt="screenshot png" src="https://github.com/user-attachments/assets/86783de8-134d-44bd-b812-7a0b06ea4db8" />


---

## What it does

Ask a business question in plain English, such as *"Why are our deliveries late, and what should we do about it?"* The agent then works through it in six steps, showing each one live:

1. **Restates the question** and picks the metric that measures it.
2. **Builds a MECE issue tree** of possible drivers: shipping mode, market, customer segment, product category, time.
3. **Tests each branch with data** and rules out the ones that don't matter.
4. **Drills into and sizes** the real drivers: the sales and profit at stake.
5. **Recommends 2–3 actions**, ranked by impact versus effort.
6. **Writes an executive summary** using the pyramid principle: the answer first.

**Example finding:** First Class orders are late 95.3% of the time against 54.8% overall, in every market. First Class promises delivery in 1 day but takes 2. That puts $5.41M of sales (14.7% of the total) at risk. The top recommendation is to reset the First Class promise to what the network actually delivers.

## How it works

- **The AI never computes numbers.** Every figure comes from tested Python functions in [`src/tools.py`](src/tools.py). An automatic check flags any number in a summary that can't be traced back to a tool result.
- **Pluggable "brains".** The planner that decides what to analyse can be swapped without touching the analysis code:
  - **Scripted rules** (default): a fixed consulting workflow with no AI model. Free, offline, and gives the same answer every time.
  - **Local LLM**: an open model (Qwen 2.5 7B) running on your own machine through [Ollama](https://ollama.com). Free and private.
- **Transparent.** Every step (tool call, result, issue tree, chart) is streamed live and saved, so any run can be replayed later.

## Run it yourself

You need **Python 3.10 or newer** and **git**. The commands below are for Windows PowerShell; macOS/Linux equivalents are noted.

### 1. Clone the repository

```bash
git clone https://github.com/vishkeru/Supply-Chain-Agent.git
cd Supply-Chain-Agent
```

### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
```

> If PowerShell says *"running scripts is disabled"*, skip activation and use `.\venv\Scripts\python.exe` in place of `python` in the commands below.

### 3. Install the requirements

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

1. Download **DataCo Smart Supply Chain for Big Data Analysis** from Kaggle:
   <https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis>
   (a free Kaggle account is needed).
2. Unzip it and put **`DataCoSupplyChainDataset.csv`** in a `data` folder at the top of the project:

```
Supply-Chain-Agent/
└── data/
    └── DataCoSupplyChainDataset.csv
```

The dataset isn't included in this repository: it's about 90 MB and is distributed by Kaggle.

### 5. (Optional) Install Ollama for the local LLM brain

1. Install Ollama from <https://ollama.com/download>.
2. Download the model (about 4.7 GB, one time only):

```bash
ollama pull qwen2.5:7b
```

Keep Ollama running and **"Local LLM (Ollama)"** appears in the app's brain dropdown. Without a graphics card, a full analysis can take several minutes.

### 6. Start the app

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. The first time, Streamlit may ask for an email address in the terminal; just press Enter. Click an example question to start.

## What you'll see, depending on your setup

| You have | The app offers |
|---|---|
| Dataset + Ollama | Live analysis with the **scripted** or **local LLM** brain, plus replays |
| Dataset only | Live analysis with the **scripted** brain, plus replays |
| No dataset (with or without Ollama) | **Replay mode**: live analysis is switched off and you can watch the saved runs in [`demo_runs/`](demo_runs), step by step, exactly as they were recorded |

Replay mode is what the hosted demo uses. The saved runs contain every tool result, so the reasoning trail, issue tree, charts and summary all replay without the dataset or any AI model.

## Project structure

```
app.py                  Streamlit app: reads events from a brain and draws them
src/tools.py            Analysis functions: the only place numbers are computed
src/brains/scripted.py  Rule-based brain (no LLM)
src/brains/ollama_brain.py  Local LLM brain (Qwen 2.5 7B via Ollama)
src/events.py           Standard event format every brain must follow
src/number_check.py     Flags numbers in a summary that no tool returned
src/run_store.py        Saves runs to demo_runs/ and replays them
demo_runs/              Saved runs used by replay mode
```

## Notes

- One row in the dataset is one **order line** (a product within an order), not a whole order.
- A branch counts as a driver only if a segment with **at least 1,000 order lines** is more than **5 percentage points** worse than average. This keeps small, noisy segments from being reported as drivers.
- Delivery times in the dataset are spread almost perfectly evenly, which suggests it is partly synthetic. Treat the findings as a demonstration of the method rather than facts about a real company.

**Built with** Python · pandas · Streamlit · Plotly · pytest · Ollama
