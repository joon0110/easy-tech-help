"""Local text analysis and reference-backed explanations."""

import streamlit as st

from easy_tech_help.analysis import DEFAULT_MODEL, LocalModelError
from easy_tech_help.config import load_settings
from easy_tech_help.rag import explain_text
from easy_tech_help.schemas import MAX_TEXT_CHARS

st.set_page_config(page_title="EasyTechHelp", page_icon="🧭", layout="centered")
st.title("EasyTechHelp")
st.write("Paste a message or alert, or describe your iPhone Wi-Fi status in English.")
st.caption(
    "Remove passwords, verification codes and account details. Your input is not saved."
)
st.info(
    "This preview explains text using local references. It does not verify who sent a message. Next-action guidance is not yet available."
)

with st.form("text_analysis"):
    text = st.text_area(
        "Message, alert or current situation",
        placeholder="Example: My iPhone is connected to Wi-Fi, but it says No Internet Connection.",
        height=180,
        max_chars=MAX_TEXT_CHARS,
    )
    submitted = st.form_submit_button("Analyze text")

if submitted:
    try:
        with st.spinner("Checking your text and local references…"):
            result = explain_text(
                text, model=load_settings().local_model or DEFAULT_MODEL
            )
    except (ValueError, LocalModelError) as exc:
        st.error(str(exc))
    else:
        st.subheader("What this describes")
        st.write(result.observation.summary)
        st.write(result.observation.uncertainty)
        if result.status == "answered":
            st.subheader("Explanation from a reference")
            st.text(result.explanation)
            st.caption(result.message)
            st.subheader("Source used for this explanation")
            for citation in result.citations:
                reference = citation.reference
                st.write(reference.title)
                st.caption(
                    "FTC original article"
                    if reference.source_type == "original_html"
                    else "Apple-based summary, not an original Apple page"
                )
                st.text(citation.quote)
                for url in reference.urls:
                    st.link_button("Open official source", url)
        else:
            st.info(result.message)
            if result.retrieved:
                with st.expander("Related references — no explanation available"):
                    for reference in result.retrieved:
                        st.write(reference.title)
                        st.caption(
                            "FTC original article"
                            if reference.source_type == "original_html"
                            else "Apple-based summary, not an original Apple page"
                        )
                        st.text(reference.excerpt)
                        for url in reference.urls:
                            st.link_button(
                                "Open official source",
                                url,
                                key=f"{reference.source_id}:{url}",
                            )
        if result.observation.signals:
            st.subheader("Evidence quoted from your text")
            # Plain text keeps untrusted links and markup from becoming controls.
            for quote in dict.fromkeys(
                item.evidence for item in result.observation.signals
            ):
                st.text(quote)
        with st.expander("Analysis details"):
            st.json(result.observation.model_dump())
