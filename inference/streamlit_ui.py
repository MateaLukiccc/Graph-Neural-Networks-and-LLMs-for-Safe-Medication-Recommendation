import streamlit as st
import os
import json
from dotenv import load_dotenv
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.parser.parser import parse_pdf
from inference.gnn_model import GNNModel
from inference.report_llm import DrugReportGenerator
from inference.main import format_prediction_report  


load_dotenv()
generator = DrugReportGenerator()

@st.cache_resource
def load_gnn_model():
    return GNNModel(model_name="safe_drug_model", dim=32, device="cpu")

st.set_page_config(page_title="Drug Recommendation Assistant", layout="wide")
st.title("Drug Recommendation Assistant")
tab1, tab2 = st.tabs(["Report Generation", "Clinical Q&A Chat"])

# ------------------------
# TAB 1: Report Generation
# ------------------------
with tab1:
    st.subheader("Upload PDF and Generate Report")

    pdf_file = st.file_uploader("Upload patient PDF", type=["pdf"])

    if pdf_file is not None:
        with st.spinner("Processing PDF..."):
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name

            admissions = parse_pdf(tmp_path)
            st.subheader("Extracted Admission Data")
            st.json(admissions)

            gnn_model = load_gnn_model()
            data = GNNModel.parse_input_to_indices(
                admissions,
                gnn_model.voc['diag_voc'],
                gnn_model.voc['pro_voc'],
                gnn_model.voc['med_voc']
            )
            predictions = gnn_model.predict(data)
            st.session_state["predictions"] = predictions

            raw_report = format_prediction_report(predictions)
            st.subheader("Prediction Report")
            st.markdown(raw_report)

            with st.spinner("Generating clinician-friendly report..."):
                structured_report = generator.generate_report(raw_report)

            st.subheader("Clinician-Friendly Report")
            try:
                report_dict = json.loads(structured_report)
                st.json(report_dict)
            except:
                st.write(structured_report)

            st.session_state["base_report"] = raw_report
            st.session_state["structured_report"] = structured_report
            st.success("Report generated. Switch to the 'Clinical Q&A Chat' tab to ask questions.")

# ------------------------
# TAB 2: Clinical Q&A Chat
# ------------------------
with tab2:
    st.subheader("Clinical Q&A")

    if "structured_report" not in st.session_state or "predictions" not in st.session_state:
        st.warning("Please first generate a report in the 'Report Generation' tab.")
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        QA_PROMPT = """
        You are a clinical pharmacology expert assisting physicians.
        You will be asked questions about drug choices, recommendations, or exclusions.

        Instructions:
        - Base reasoning on pharmacology, comorbidities, and prescribing guidelines.
        - Use patient's ICD9 codes, procedures, ATC codes, and model predictions as context.
        - Expand on why a drug might be suggested or omitted — do not just repeat the report.
        - Highlight ICD/ATC codes in **bold** when referenced.
        - Highlight prediction confidence (high/medium/low) when relevant.
        - Avoid making direct prescriptions; instead, provide reasoning and considerations.
        """

        for msg in st.session_state["messages"]:
            role = "Clinician" if msg["role"] == "user" else "Expert Assistant"
            st.markdown(f"**{role}:** {msg['content']}")

        user_question = st.chat_input("Ask a question about the drug recommendations...")
        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})

            conversation = [
                {"role": "system", "content": QA_PROMPT},
                {
                    "role": "user",
                    "content": f"""
Patient structured report:
{st.session_state['structured_report']}

Raw model predictions (IDs, drug names, probabilities):
{json.dumps(st.session_state['predictions'], indent=2)}
"""
                },
            ] + st.session_state["messages"]

            with st.spinner("Thinking like a medical expert..."):
                response = generator.client.chat.completions.create(
                    model=generator.REPORT_PARSE_MODEL,
                    messages=conversation
                )
                answer = response.choices[0].message.content

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.markdown(f"**Expert Assistant:** {answer}")