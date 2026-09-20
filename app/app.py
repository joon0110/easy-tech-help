"""Streamlit entry point for the EasyTechHelp foundation."""

import streamlit as st

st.set_page_config(page_title="EasyTechHelp", page_icon="🧭", layout="centered")

st.title("EasyTechHelp")
st.write("Clear next steps when a phone screen is confusing.")

st.info("This is an early preview. Screenshot analysis is not available yet.")

st.subheader("What EasyTechHelp is being built to help with")
st.markdown(
    """
    - Pop-ups and security warnings
    - Text messages and emails
    - Wi-Fi and connectivity settings
    """
)
