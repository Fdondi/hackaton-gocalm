import io
from pathlib import Path
from typing import Dict, List

import streamlit as st
from pypdf import PdfReader

try:
    from .infer import infer_text, load_inference_bundle
except ImportError:
    from multihead_pii.infer import infer_text, load_inference_bundle


def _extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Dict] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        pages.append({"page": idx, "text": text})
    return pages


def _build_redacted_text(text: str, redactions: List[Dict]) -> str:
    if not redactions:
        return text
    output = text
    for row in sorted(redactions, key=lambda item: item["start"], reverse=True):
        start = int(row["start"])
        end = int(row["end"])
        label = str(row.get("label", "PII"))
        replacement = f"[REDACTED:{label}]"
        output = output[:start] + replacement + output[end:]
    return output


@st.cache_resource(show_spinner=False)
def _get_bundle(checkpoint_path: str, device: str) -> Dict:
    return load_inference_bundle(checkpoint_path=checkpoint_path, device=device)


def main() -> None:
    st.set_page_config(page_title="PII PDF Redaction Demo", layout="wide")
    st.title("PII PDF Redaction Demo")
    st.caption("Upload a PDF, extract text, run the trained model, and inspect redactions.")

    default_ckpt = str(Path("outputs_multihead/multihead_model.pt"))
    checkpoint_path = st.text_input("Checkpoint path", value=default_ckpt)
    device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
    uploaded_pdf = st.file_uploader("Select a PDF", type=["pdf"])

    if uploaded_pdf is None:
        st.info("Choose a PDF to start.")
        return

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    with st.spinner("Reading PDF..."):
        pages = _extract_pdf_pages(uploaded_pdf.read())
    if not pages:
        st.warning("No extractable text found in this PDF.")
        return

    with st.spinner("Loading model..."):
        bundle = _get_bundle(checkpoint_path=str(checkpoint), device=device)

    all_components: List[Dict] = []
    total = len(pages)
    progress = st.progress(0.0)
    for index, page in enumerate(pages, start=1):
        result = infer_text(page["text"], bundle)
        for redaction in result["redactions"]:
            row = dict(redaction)
            row["page"] = page["page"]
            all_components.append(row)
        page["result"] = result
        progress.progress(index / total)
    progress.empty()

    st.subheader("Detected redacted components")
    if not all_components:
        st.success("No redactions detected.")
    else:
        st.dataframe(all_components, use_container_width=True)

    st.subheader("Redacted text preview")
    for page in pages:
        result = page["result"]
        with st.expander(f"Page {page['page']}", expanded=False):
            if result.get("windowed", False):
                st.info(f"Processed in {result.get('window_count', 1)} model windows.")
            st.text_area(
                label=f"Page {page['page']} redacted text",
                value=_build_redacted_text(page["text"], result["redactions"]),
                height=220,
                label_visibility="collapsed",
            )


if __name__ == "__main__":
    main()
