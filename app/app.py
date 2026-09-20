"""Streamlit entry point for the EasyTechHelp foundation."""

import streamlit as st

st.set_page_config(page_title="EasyTechHelp", page_icon="🧭", layout="centered")

st.title("EasyTechHelp")
st.write("Clear next steps when a phone screen is confusing.")

st.info(
    "This is an early preview. Local screenshot analysis is available from the command line; upload is not connected here yet."
)

st.subheader("What EasyTechHelp is being built to help with")
st.markdown(
    """
    - Safari pop-ups and security warnings on iPhone
    - Text messages and emails on iPhone
    - iPhone Wi-Fi and connectivity settings
    """
)
