# see README for usage
from pathlib import Path
import os, json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

try:
    import openai
except Exception:
    openai = None

EMBED_MODEL_OPENAI = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\\r\\n", "\\n")
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception:
        return ""

def load_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def load_text_from_pdf(path: Path) -> str:
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            texts = []
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    continue
            return "\\n".join(texts)
    except Exception:
        return ""

from dataclasses import dataclass
@dataclass
class KBItem:
    source: str
    text: str
    chunks: List[str]

def build_knowledge_items(folder: str) -> List[KBItem]:
    p = Path(folder)
    items = []
    if not p.exists():
        return items
    for fp in p.glob("**/*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in [".docx"]:
            text = load_text_from_docx(fp)
        elif fp.suffix.lower() in [".txt", ".md"]:
            text = load_text_from_txt(fp)
        elif fp.suffix.lower() in [".pdf"]:
            text = load_text_from_pdf(fp)
        else:
            continue
        if not text or len(text.strip()) < 50:
            continue
        chunks = chunk_text(text)
        items.append(KBItem(source=str(fp), text=text, chunks=chunks))
    return items

def get_embeddings(texts):
    if OPENAI_KEY and openai:
        openai.api_key = OPENAI_KEY
        results = []
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp = openai.Embedding.create(input=batch, model=EMBED_MODEL_OPENAI)
            for r in resp["data"]:
                results.append(r["embedding"])
        return results
    else:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embs.tolist()
        except Exception as e:
            raise RuntimeError("No embedding backend available. Install openai and set OPENAI_API_KEY or install sentence-transformers.") from e

def build_faiss_index(embeddings, dim: int):
    try:
        import faiss, numpy as np
    except Exception as e:
        raise RuntimeError("faiss or numpy not available. Install faiss-cpu and numpy.") from e
    xb = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    return index

from dataclasses import dataclass
import numpy as np
@dataclass
class VectorKB:
    items: List[KBItem]
    embeddings: List[List[float]]
    index: object
    meta: List[Tuple[int,int,str]]

def build_vector_kb(folder: str = "data_sources") -> VectorKB:
    items = build_knowledge_items(folder)
    passages = []
    meta = []
    for i,item in enumerate(items):
        for j,ch in enumerate(item.chunks):
            passages.append(ch)
            meta.append((i,j,item.source))
    if not passages:
        return VectorKB(items=[], embeddings=[], index=None, meta=[])
    embeddings = get_embeddings(passages)
    dim = len(embeddings[0])
    index = build_faiss_index(embeddings, dim)
    return VectorKB(items=items, embeddings=embeddings, index=index, meta=meta)

def retrieve(kb: VectorKB, query: str, top_k: int = 4):
    if not kb or not getattr(kb, "index", None):
        return []
    emb_q = get_embeddings([query])[0]
    xq = np.array([emb_q]).astype("float32")
    D, I = kb.index.search(xq, top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(kb.meta):
            continue
        item_idx, chunk_idx, src = kb.meta[idx]
        chunk_text = kb.items[item_idx].chunks[chunk_idx]
        results.append({"source": src, "chunk": chunk_text, "score": float(D[0][rank])})
    return results

def analyze_with_llm(query_text: str, retrieved: List[Dict], max_tokens: int = 800) -> List[Dict]:
    context = ""
    for i,r in enumerate(retrieved, start=1):
        src = r.get("source","")
        snippet = r.get("chunk","").strip().replace("\\n", " ")
        context += f"[{i}] Source: {src}\\n{snippet}\\n\\n"
    system = "You are an expert legal assistant familiar with ADGM company formation regulations. Analyze the provided document text for ADGM compliance issues. For each issue, provide: section (approximate), issue description, severity (Low/Medium/High), suggestion to fix, and which retrieved source(s) support this finding (cite by [n]). If unsure, say 'no issues found.'"
    user = f"""Document Text:
\"{query_text[:6000]}\"

Relevant ADGM reference passages (use for citations):
{context}

Instructions: Provide a JSON array where each element is an object with keys:
- section
- issue
- severity
- suggestion
- citations (list of integers referring to the passages above)
Return only valid JSON.
"""

    if OPENAI_KEY and openai:
        try:
            openai.api_key = OPENAI_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if openai else "gpt-4o-mini",
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            out = resp["choices"][0]["message"]["content"].strip()
            try:
                return json.loads(out)
            except Exception:
                import re
                m = re.search(r"(\\[.*\\])", out, re.S)
                if m:
                    return json.loads(m.group(1))
                else:
                    return [{"section":"N/A","issue":"LLM output parsing failed","severity":"Low","suggestion":out,"citations":[]}]
        except Exception as e:
            return [{"section":"N/A","issue":f"OpenAI call failed: {e}","severity":"Low","suggestion":"Check OPENAI_API_KEY and network.","citations":[]}]
    else:
        low = query_text.lower()
        findings = []
        if "federal" in low or "federal courts" in low or "u.a.e." in low:
            cites = [i+1 for i,r in enumerate(retrieved) if "adgm" in r.get("chunk","").lower() or "jurisdiction" in r.get("chunk","").lower()]
            findings.append({"section":"Jurisdiction Clause (heuristic)","issue":"Jurisdiction clause may reference UAE Federal Courts instead of ADGM.","severity":"High","suggestion":"Replace jurisdiction references with ADGM Courts or specify ADGM jurisdiction.","citations":cites})
        if "signature" not in low and "signed by" not in low:
            cites = [i+1 for i,r in enumerate(retrieved) if "signature" in r.get("chunk","").lower() or "execution" in r.get("chunk","").lower()]
            findings.append({"section":"Execution Block","issue":"Possible missing or improperly formatted signatory section.","severity":"Medium","suggestion":"Ensure an execution/signature block with names, titles, and dates is present.","citations":cites})
        if "ubo" not in low and "ultimate beneficial owner" not in low:
            cites = [i+1 for i,r in enumerate(retrieved) if "ubo" in r.get("chunk","").lower() or "beneficial owner" in r.get("chunk","").lower()]
            findings.append({"section":"UBO Declaration","issue":"No UBO declaration detected.","severity":"High","suggestion":"Provide a UBO Declaration Form as required by ADGM.","citations":cites})
        return findings

def load_knowledge_base(folder: str = "data_sources"):
    from dataclasses import dataclass
    # lazy import to avoid heavy startup cost
    return build_vector_kb(folder)

def query_rag_for_issues(doc_text, kb=None):
    """Wrapper to retrieve from KB and run analysis."""
    if not kb:
        return [{"section": "N/A", "issue": "No knowledge base loaded", "severity": "Low",
                 "suggestion": "Click 'Build KB / Index' first.", "citations": []}]
    retrieved = retrieve(kb, doc_text, top_k=4)
    return analyze_with_llm(doc_text, retrieved)


def checklist_compare(uploaded_types, required_types):
    """Compare uploaded doc types against required types."""
    missing = [r for r in required_types if r not in uploaded_types]
    present = [u for u in uploaded_types if u in required_types]
    return {"missing": missing, "present": present}
