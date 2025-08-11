# 🏛 AI-Powered ADGM Corporate Agent

An AI-powered compliance assistant that reviews corporate legal documents (`.docx`) for **ADGM (Abu Dhabi Global Market)** compliance using **Retrieval-Augmented Generation (RAG)** and a simple **Streamlit** UI.

This project was built as part of the **AI Engineer Intern Assessment** for 2Cents Capital.

---

## 📌 Features

- **Upload & Parse** `.docx` files (AoA, MoA, Resolutions, UBO, etc.)
- **Checklist Verification** against official ADGM requirements
- **RAG-powered Compliance Review** with legal citations
- **Inline Comments** in reviewed documents
- **Structured JSON Output** summarizing:
  - Process detected
  - Missing documents
  - Compliance issues (with severity & suggestions)
  - Citations to relevant ADGM passages
- **Download Outputs**:
  - Annotated `.docx`
  - JSON analysis

---

## 🛠 Tech Stack

- **Python 3.9+**
- **Streamlit** – interactive UI
- **python-docx** – document parsing & annotation
- **FAISS / Sentence-Transformers** – vector search for RAG
- **OpenAI API** – AI compliance analysis (optional but recommended)
- **python-dotenv** – environment variable management

---

## 📂 Project Structure

corporate_agent_project/
│
├── app.py # Streamlit UI
├── utils/
│ ├── docx_utils.py # Document parsing & comment insertion
│ ├── rag_utils.py # Knowledge base & AI analysis logic
│
├── data_sources/ # ADGM reference docs (PDF/DOCX)
├── sample_docs/ # Sample input documents
├── outputs/ # Generated annotated docs & JSON
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env # OpenAI API key (not committed to GitHub)

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Vigneshv3/adgm-corporate-agent-ai.git
cd adgm-corporate-agent-ai
2️⃣ Create a virtual environment
Windows (PowerShell)

powershell
Copy
Edit
python -m venv venv
venv\Scripts\activate
Mac/Linux

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Add your OpenAI API key
Create a .env file in the project root:

env
Copy
Edit
OPENAI_API_KEY=your_openai_api_key_here
💡 Without an API key, the app will use local embeddings and heuristics.

5️⃣ Add ADGM reference documents
Download official ADGM checklists & templates from the provided sources and place them into:

Copy
Edit
data_sources/
6️⃣ Run the app
bash
Copy
Edit
streamlit run app.py
The app will open at http://localhost:8501.

📊 Usage Flow
Build Knowledge Base

Click the "Build KB / Index" button in the app to index all files in data_sources/.

Upload Documents

Upload your .docx files for review.

AI Analysis

The system checks compliance, flags issues, and adds comments.

Download Outputs

Annotated .docx

analysis_output.json

📸 Example Output
Sample JSON:

json
Copy
Edit
{
  "process": "Company Incorporation",
  "documents_uploaded": 4,
  "required_documents": 5,
  "missing_document": "Register of Members and Directors",
  "issues_found": [
    {
      "document": "Articles of Association",
      "section": "Clause 3.1",
      "issue": "Jurisdiction clause does not specify ADGM",
      "severity": "High",
      "suggestion": "Update jurisdiction to ADGM Courts.",
      "citations": [2, 5]
    }
  ]
}

Author : Vignesh Deenadayal

📜 License
This project is for assessment purposes only. ADGM references belong to their respective owners.

🙌 Acknowledgements
ADGM Official Website

Streamlit

FAISS

OpenAI

