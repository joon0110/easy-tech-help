"""Text input and validated observations for the local V1 foundation."""

import streamlit as st

from easy_tech_help.analysis import DEFAULT_MODEL, LocalModelError, analyze_text
from easy_tech_help.config import load_settings
from easy_tech_help.schemas import MAX_TEXT_CHARS

st.set_page_config(page_title="EasyTechHelp", page_icon="🧭", layout="centered")
st.title("EasyTechHelp")
st.write("Paste a message or alert, or describe your iPhone Wi-Fi status in English.")
st.caption(
    "Remove passwords, verification codes and account details. Your input is not saved."
)
st.info(
    "This preview analyzes text. Cautions, next steps and source guidance are coming later."
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
        with st.spinner("Analyzing your text…"):
            result = analyze_text(
                text, model=load_settings().local_model or DEFAULT_MODEL
            )
    except (ValueError, LocalModelError) as exc:
        st.error(str(exc))
    else:
        st.subheader("What this describes")
        st.write(result.summary)
        st.write(result.uncertainty)
        if result.signals:
            st.subheader("Evidence quoted from your text")
            # Plain text keeps untrusted links and markup from becoming controls.
            for quote in dict.fromkeys(item.evidence for item in result.signals):
                st.text(quote)
        with st.expander("Analysis details"):
            st.json(result.model_dump())
