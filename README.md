# 🔍 Tavily-Gemma Research Helper

**Tavily‑Gemma Research Helper** is a smart, local‑first research assistant that helps you analyze any topic with real‑time web data and step‑by‑step research guidance powered by LangGraph.

It’s more than just a summarizer—it’s a full AI research workflow designed to teach critical thinking and show how insights are generated.

This app helps you understand how to approach a topic from multiple angles. It gives you initial leads, suggests exploratory paths, and offers different lenses through which to run your brain around the subject. By using this tool regularly, you'll develop a sharper research mindset and learn to think critically and creatively about any topic you explore.
Over time, it turns research from a task into a skill — and curiosity into a method.

---

## 🔍 What It Does

This app helps you:

* Plan research workflows using a LangGraph agent
* Query a **local LLM** (Gemma 3B) via LM Studio
* Pull live search results using the Tavily API
* Review a clean, transparent research output (with reasoning path!)

---

## 🧠 Tech‑Powered Workflow

Each part of the app is purpose‑built for clarity and flexibility:

### 🧭 Research Flow Orchestration (LangGraph)

LangGraph powers the structured research pipeline: planning → searching → synthesis.

### 🤖 Local Reasoning via LM Studio (google/gemma-3-1b)

**LLM used:** `google/gemma-3-1b`, served via LM Studio’s OpenAI‑compatible API mode.
Why this model? It’s fast, instruction‑tuned, and excels at structured thought processes.

### 🌐 Search & Data via Tavily API

Live web results come from Tavily Search, curated and formatted into source‑rich summaries.

---

## 🛠️ Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/aTh1ef/tavily-gemma-researcher.git
cd tavily-gemma-researcher
```

### Step 2: Install Dependencies

Ensure you have **Python 3.10+**, then:

```bash
pip install -r requirements.txt
```

---

## 🤖 LM Studio Setup

To run gemma-3-1b locally:

1. Install [LM Studio](https://lmstudio.ai/)
2. In LM Studio → **Models** tab, search for **`google/gemma-3-1b`**
3. Download the GGUF (chat/instruct) version
4. Go to **Server** tab:

   * Toggle **OpenAI‑compatible API**
   * Ensure it’s running at `http://localhost:1234/v1`

---

## 🔐 Streamlit Secrets Configuration

Create a `.streamlit/secrets.toml` file:

```toml
[tavily]
api_key = "your_tavily_api_key"

[llmstudio]
api_base = "http://localhost:1234/v1"
api_key = "lm-studio"
```

---

## ✅ Run the App

```bash
streamlit run app.py
```
---

## 🔁 How It Works

Once the app is running:

1. 📝 Enter your research topic
2. 🚀 Click **Start Research**

Under the hood:

* LangGraph breaks your topic into sub‑questions
* LM Studio (using **google/gemma-3-1b**) frames and expands ideas locally
* Tavily fetches up‑to‑date search results
* Results are organized by theme and credibility
* A clean research summary is synthesized
* You see all steps of the reasoning chain so you learn *how* the AI arrived at its conclusions

---

## 💻 What You Need to Do

Before launching:

* ✅ Get your **Tavily API Key**
* ✅ Install **LM Studio**
* ✅ Download the **`google/gemma-3-1b`** GGUF model
* ✅ Enable **OpenAI‑compatible API** in LM Studio (`http://localhost:1234/v1`)
* ✅ Add your `.streamlit/secrets.toml` config
* ✅ Run the app and input your topic

---

## 📚 Use Cases

* Investigating controversial claims
* Breaking down complex questions
* Sourcing reliable information from the web
* Practicing the research process with AI assistance


---

## 🤝 Contributing

Ideas, bugs, or feature requests? Open an issue or pull request—collaborators welcome!

---

## 🧠 Learn the Process, Not Just the Output

Tavily‑Gemma Research Helper doesn’t just tell you what’s true—it walks you through *how* it figured it out.
Perfect for students, analysts, and anyone who wants AI that *teaches*, not just *tells*.
