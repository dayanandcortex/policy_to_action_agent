# 🧠 Policy-to-Action Agent

An end-to-end autonomous AI agent system that converts unstructured policy documents (PDF/TXT/MD) into structured rules, verified decisions, and actionable insights using LLMs.

## 🚀 Overview

This project builds a multi-agent pipeline that:
- Reads policy documents
- Extracts structured entities and rules
- Verifies rules against source text
- Generates decisions
- Critiques and refines outputs

## 🏗️ Architecture

High-Level Flow:

Input → Reader → Classifier → Extractor → Verifier → Decision → Critic

## 📂 Project Structure

```
policy_to_action_agent/
├── app/
│   ├── agents/
│   ├── ui/
│   ├── config.py
│   ├── graph.py
│   ├── schemas.py
│   ├── state.py
│   └── utils.py
├── run.py
├── .env
└── README.md
```

## ⚙️ Setup

```bash
conda create -n p2a python=3.11
conda activate p2a
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run app/ui/streamlit_app.py
```

## 📊 Output

Returns structured JSON with:
- document_type
- extracted_entities
- verified_rules
- final_decision
- confidence

## Evaluation

Compare the multi-agent pipeline against a one-shot LLM baseline:

```bash
python -m app.eval.run_evaluation app/data/final-sbc-and-clfs-from-naic.pdf
```

The evaluation writes:
- `outputs/eval/multi_agent_result.json`
- `outputs/eval/baseline_result.json`
- `outputs/eval/comparison_report.json`

The comparison report includes quantitative metrics such as extracted rule count,
supported verified rule count, support rate, evidence coverage, page-number
coverage, average confidence, runtime, and a short qualitative summary.
