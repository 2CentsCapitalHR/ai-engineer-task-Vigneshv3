import streamlit as st
from utils.docx_utils import parse_docx, insert_comments_and_save, detect_document_type
from utils.rag_utils import load_knowledge_base, query_rag_for_issues, checklist_compare
import json
from pathlib import Path

st.set_page_config(page_title="ADGM Corporate Agent — RAG Enhanced", layout="wide")

st.title("ADGM Corporate Agent — RAG Enhanced")
st.markdown("Upload `.docx` files and run an AI-powered ADGM compliance review. Make sure you created a `.env` with `OPENAI_API_KEY` (optional) and placed ADGM reference docs in `data_sources/`.")

col1, col2 = st.columns([1,3])
with col1:
    st.header("Knowledge Base")
    kb_folder = st.text_input("KB folder", value="data_sources")
    if st.button("Build KB / Index"):
        with st.spinner("Building KB (this may take a while for large datasets)..."):
            kb = load_knowledge_base(kb_folder)
            st.session_state["kb"] = kb
            st.success(f"KB built. Documents: {len(kb.items)}")
    if st.button("Clear KB"):
        st.session_state["kb"] = None
        st.success("Cleared KB")

with col2:
    uploaded = st.file_uploader("Upload one or more .docx files", accept_multiple_files=True, type=["docx"])
    if uploaded:
        st.info(f"Received {len(uploaded)} file(s). Processing...")
        Path("outputs").mkdir(exist_ok=True)
        docs = []
        for f in uploaded:
            out_path = Path("outputs") / f.name
            with open(out_path, "wb") as wf:
                wf.write(f.getbuffer())
            parsed = parse_docx(out_path)
            doc_type = detect_document_type(parsed)
            docs.append({"path": str(out_path), "name": f.name, "text": parsed, "type": doc_type})

        process = "Unknown"
        types = set(d["type"] for d in docs)
        if any("Articles" in t for t in types) or any("Memorandum" in t for t in types):
            process = "Company Incorporation"

        required = ["Articles of Association", "Memorandum of Association", "Incorporation Application Form", "UBO Declaration Form", "Register of Members and Directors"]
        uploaded_types = [d["type"] for d in docs]
        checklist = checklist_compare(uploaded_types, required)

        issues = []
        kb = st.session_state.get("kb", None)
        for d in docs:
            with st.spinner(f"Analyzing {d['name']} ..."):
                findings = query_rag_for_issues(d["text"], kb=kb)
                for f in findings:
                    f["document"] = d["name"]
                issues.extend(findings)

        output = {
            "process": process,
            "documents_uploaded": len(docs),
            "required_documents": len(required),
            "missing_documents": checklist["missing"],
            "issues_found": issues,
        }

        st.subheader("Summary")
        st.json(output)

        out_json_path = Path("outputs") / "analysis_output.json"
        with open(out_json_path, "w", encoding='utf-8') as jf:
            json.dump(output, jf, indent=2, ensure_ascii=False)

        st.success("Saved structured JSON to outputs/analysis_output.json")

        demo_doc = docs[0]
        annotated_path = Path("outputs") / f"annotated_{Path(demo_doc['name']).stem}.docx"
        insert_comments_and_save(demo_doc["path"], annotated_path, issues)

        with open(annotated_path, "rb") as bf:
            st.download_button("Download annotated .docx", data=bf, file_name=annotated_path.name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        with open(out_json_path, "rb") as jf:
            st.download_button("Download analysis JSON", data=jf, file_name="analysis_output.json", mime="application/json")

    else:
        st.info("No files uploaded yet. Build KB first and then upload files.")