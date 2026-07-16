# app.py
# The actual website. Run locally with: streamlit run app.py

import streamlit as st
from agent_core import run_claim

st.set_page_config(page_title="ClaimGraph", page_icon="📋")

st.title("ClaimGraph")
st.caption("Insurance Claims Adjudication Agent")

claim_text = st.text_area(
    "Describe the claim:",
    placeholder="e.g. My car was damaged by flooding, do I have coverage?",
    height=100,
)

if st.button("Submit claim", type="primary"):
    if not claim_text.strip():
        st.warning("Please enter a claim description first.")
    else:
        with st.spinner("Agent is reviewing the claim..."):
            result = run_claim(claim_text)

        decision = result["decision"]
        if decision == "approve":
            st.success(f"Decision: {decision.upper()}")
        elif decision == "deny":
            st.error(f"Decision: {decision.upper()}")
        else:
            st.warning(f"Decision: {decision.upper()}")

        st.write("**Reasoning:**")
        st.write(result["reasoning"])
