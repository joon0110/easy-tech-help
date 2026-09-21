"""A readable, local text-help interface built around one task at a time."""

from pathlib import Path

import streamlit as st

from easy_tech_help.analysis import DEFAULT_MODEL, LocalModelError
from easy_tech_help.config import load_settings
from easy_tech_help.guidance import guide_text
from easy_tech_help.safety import displayable_reference
from easy_tech_help.schemas import MAX_TEXT_CHARS, validate_input

EXAMPLES = {
    "Text message": "I received a text asking me to enter my account password through a link. What should I do?",
    "Popup or alert": "A Safari popup says my iPhone has a virus and tells me to call a support number.",
    "Wi-Fi problem": "My iPhone is connected to Wi-Fi, but it says No Internet Connection.",
}
RESULT_TITLES = {
    "attention": "Pause and check this request",
    "check_source": "Check where this came from",
    "connection_check": "Let's check your connection",
    "uncertain": "A little more context would help",
    "no_specific_warning": "No specific warning signs identified",
}


def start_over(example: str = "") -> None:
    st.session_state.situation = example
    st.session_state.pop("product", None)
    st.session_state.pop("checked_text", None)
    st.session_state.pop("check_error", None)


def show_result(product) -> None:
    result, guidance = product.analysis, product.guidance
    st.html('<hr class="eth-rule"><p class="eth-result-label">YOUR RESULT</p>')
    st.header(RESULT_TITLES[guidance.level])
    st.write(result.observation.summary)
    with st.container(key="assessment"):
        if guidance.level in {"attention", "check_source"}:
            st.warning(guidance.summary)
        else:
            st.info(guidance.summary)
    st.subheader("Next steps")
    with st.container(key="next_steps"):
        for index, action in enumerate(guidance.next_actions, 1):
            st.text(f"{index}. {action.text}")
    if guidance.avoid:
        st.subheader("What to avoid")
        for item in guidance.avoid:
            st.text(item)
    if guidance.cautions:
        with st.expander("Things to keep in mind"):
            for caution in guidance.cautions:
                st.text(caution)
    # Keep the checked text distinct from a subsequently edited input draft.
    with st.expander("Text used for this check"):
        st.text(st.session_state.checked_text)
        st.caption(result.observation.uncertainty)
        if result.observation.signals:
            st.write("Parts that informed this guidance")
            for quote in dict.fromkeys(s.evidence for s in result.observation.signals):
                st.text(quote)
    with st.expander("Sources and explanation"):
        explanation = displayable_reference(result)
        if explanation:
            st.text(explanation)
            st.caption(result.message)
        elif result.status == "answered":
            st.caption(
                "Use the reviewed next steps above. You can read the full reference through its official link below."
            )
        else:
            st.info(result.message)
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
                        "Read on FTC.gov"
                        if source.source_type == "original_html"
                        else "Read on Apple Support",
                        url,
                        key=f"{source.document_id}:{url}",
                    )
    with st.expander("Ask family for help", key="family"):
        st.write(
            "Consider asking someone you trust to review this with you."
            if product.family_handoff.suggested
            else "Still unsure? Share this summary with someone you trust."
        )
        st.caption(
            "Your original text and personal details are left out of this summary."
        )
        st.code(product.family_handoff.text, language=None, wrap_lines=True)
        st.caption(
            "Use the copy icon at the top right of the summary, or save a text file below. Review it before sharing. Nothing is sent automatically."
        )
        st.download_button(
            "Save summary (.txt)",
            product.family_handoff.text,
            file_name="easytechhelp-family-summary.txt",
            mime="text/plain",
            on_click="ignore",
        )
    st.button("Start a new check", key="start_over", on_click=start_over)


st.set_page_config(page_title="EasyTechHelp", page_icon="◦", layout="centered")
st.html(Path(__file__).with_name("styles.css"))
st.html(
    '<header class="eth-header">'
    '<span class="eth-brand">EasyTechHelp</span>'
    '<span class="eth-local"><span class="eth-dot" aria-hidden="true"></span>'
    "Runs on your device</span></header>"
    '<section class="eth-hero">'
    '<p class="eth-eyebrow">A LITTLE CLARITY</p>'
    "<h1>What’s on your phone?</h1>"
    '<p class="eth-intro">Make sense of a message, an unexpected alert, or an iPhone Wi-Fi problem. One step at a time.</p>'
    "</section>"
)

with st.form("text_analysis", border=False):
    text = st.text_area(
        "Paste the text or describe what you see",
        placeholder="For example: My iPhone is connected to Wi-Fi, but it says No Internet Connection.",
        height=190,
        max_chars=MAX_TEXT_CHARS,
        key="situation",
    )
    st.caption("Leave out passwords, verification codes and account details.")
    submitted = st.form_submit_button("Help me understand →", type="primary")

if submitted:
    st.session_state.pop("product", None)
    st.session_state.pop("checked_text", None)
    st.session_state.pop("check_error", None)
    try:
        validate_input(text)
        with st.spinner("Reading your text and checking the references…"):
            st.session_state.product = guide_text(
                text, model=load_settings().local_model or DEFAULT_MODEL
            )
        st.session_state.checked_text = text
    except ValueError as exc:
        st.session_state.check_error = str(exc)
    except LocalModelError:
        st.session_state.check_error = (
            "The check couldn't finish. Your text is still here, so you can try again. "
            "If this keeps happening, ask the person who set up EasyTechHelp to check the local model."
        )

if error := st.session_state.get("check_error"):
    st.error(error)
if product := st.session_state.get("product"):
    show_result(product)
else:
    st.caption("Not sure where to start? Try an example.")
    with st.container(horizontal=True, gap="small", key="examples"):
        for label, example in EXAMPLES.items():
            st.button(
                label, key=f"example_{label}", on_click=start_over, args=(example,)
            )

st.html(
    '<footer class="eth-footer">English text only. Your text stays on this device and is not saved to a file.<br>'
    "Guidance can be mistaken. EasyTechHelp cannot verify a sender or guarantee that a network is safe.</footer>"
)
