"""Local text analysis and reference-backed explanations."""

import streamlit as st

from easy_tech_help.analysis import DEFAULT_MODEL, LocalModelError
from easy_tech_help.config import load_settings
from easy_tech_help.guidance import guide_text
from easy_tech_help.safety import displayable_reference
from easy_tech_help.schemas import MAX_TEXT_CHARS

st.set_page_config(page_title="EasyTechHelp", page_icon="🧭", layout="centered")
st.title("EasyTechHelp")
st.write("Paste a message or alert, or describe your iPhone Wi-Fi status in English.")
st.caption(
    "Remove passwords, verification codes and account details. Your input is not saved."
)
st.info(
    "This preview suggests cautious next steps using local references. It cannot verify a sender or guarantee that a network is safe."
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
            product = guide_text(
                text, model=load_settings().local_model or DEFAULT_MODEL
            )
    except (ValueError, LocalModelError) as exc:
        st.error(str(exc))
    else:
        result, guidance = product.analysis, product.guidance
        st.subheader("What this describes")
        st.write(result.observation.summary)
        st.write(result.observation.uncertainty)
        if guidance.level in {"attention", "check_source"}:
            st.warning(guidance.summary)
        else:
            st.info(guidance.summary)
        if guidance.cautions:
            st.subheader("Things to keep in mind")
            for caution in guidance.cautions:
                st.text(caution)
        st.subheader("Next steps")
        for index, action in enumerate(guidance.next_actions, 1):
            st.text(f"{index}. {action.text}")
        if guidance.avoid:
            st.subheader("What to avoid")
            for item in guidance.avoid:
                st.text(item)
        explanation = displayable_reference(result)
        if explanation:
            st.subheader("What the reference says")
            st.text(explanation)
            st.caption(result.message)
        elif result.status == "answered":
            st.caption(
                "Use the reviewed next steps above. The full reference is available through its official link below."
            )
        else:
            st.info(result.message)
        # Source titles and links can be read without displaying unreviewed
        # procedural passages that bypass the next-action policy.
        sources = {
            a.evidence.document_id: a.evidence
            for a in guidance.next_actions
            if a.evidence
        }
        for reference in result.retrieved:
            sources.setdefault(reference.document_id, reference)
        if sources:
            st.subheader("References")
            for source in sources.values():
                st.write(source.title)
                st.caption(
                    "FTC original article"
                    if source.source_type == "original_html"
                    else "Apple-based summary, not an original Apple page"
                )
                for url in source.urls:
                    st.link_button(
                        "Open official source", url, key=f"{source.document_id}:{url}"
                    )
        if result.observation.signals:
            st.subheader("Evidence quoted from your text")
            # Plain text keeps untrusted links and markup from becoming controls.
            for quote in dict.fromkeys(
                item.evidence for item in result.observation.signals
            ):
                st.text(quote)
        with st.expander("Analysis details"):
            st.json(
                {
                    "observation": result.observation.model_dump(),
                    "policy_flags": guidance.flags,
                    "action_ids": [a.id for a in guidance.next_actions],
                    "action_evidence": guidance.evidence_status,
                }
            )
