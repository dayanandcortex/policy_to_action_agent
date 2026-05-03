import json
import sys
import tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import get_max_revisions
from app.graph import build_graph
from app.state import AgentState

st.set_page_config(page_title="Policy-to-Action Agent", layout="wide")
st.title("Policy-to-Action Agent")
st.caption("PDF-first autonomous policy analysis with Ollama")

uploaded_file = st.file_uploader(
    "Upload a policy document",
    type=["pdf", "txt", "md"],
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if st.button("Run Analysis"):
        with st.spinner("Running agent..."):
            app = build_graph()

            initial_state: AgentState = {
                "input_path": temp_path,
                "raw_text": "",
                "page_texts": [],
                "chunks": [],
                "document_type": "",
                "extracted_entities": {},
                "extracted_rules": [],
                "verified_rules": [],
                "ambiguities": [],
                "final_decision": {},
                "critic_feedback": "",
                "confidence": 0.0,
                "revision_count": 0,
                "max_revisions": get_max_revisions(),
            }

            result = app.invoke(initial_state)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Document Type")
            st.write(result["document_type"])

            st.subheader("Extracted Entities")
            st.json(result["extracted_entities"])

            st.subheader("Ambiguities")
            st.json(result["ambiguities"])

        with col2:
            st.subheader("Final Decision")
            st.json(result["final_decision"])

            st.subheader("Confidence")
            st.write(result["confidence"])

            st.subheader("Critic Feedback")
            st.write(result["critic_feedback"])

        st.subheader("Extracted Rules")
        st.json(result["extracted_rules"])

        st.subheader("Verified Rules")
        st.json(result["verified_rules"])

        output_json = json.dumps(
            {
                "document_type": result["document_type"],
                "extracted_entities": result["extracted_entities"],
                "extracted_rules": result["extracted_rules"],
                "verified_rules": result["verified_rules"],
                "ambiguities": result["ambiguities"],
                "final_decision": result["final_decision"],
                "critic_feedback": result["critic_feedback"],
                "confidence": result["confidence"],
            },
            indent=2,
        )

        st.download_button(
            label="Download Result JSON",
            data=output_json,
            file_name="policy_analysis_result.json",
            mime="application/json",
        )