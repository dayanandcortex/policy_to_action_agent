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

## ⚙️ Setup

conda create -n p2a python=3.11
conda activate p2a
pip install -r requirements.txt

## ▶️ Run

streamlit run app/ui/streamlit_app.py

## 📊 Output

Returns structured JSON with:
- document_type
- extracted_entities
- verified_rules
- final_decision
- confidence

