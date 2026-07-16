import streamlit as st
from agent_core import run_claim

st.set_page_config(page_title="ClaimGraph", page_icon="📋", layout="centered")

st.title("ClaimGraph")
st.caption("Insurance Claims Adjudication Agent")

# Input field
user_input = st.text_area(
    "Describe the claim:", 
    placeholder="e.g. My car was damaged by flooding, do I have coverage?"
)

if st.button("Submit claim", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter claim details before submitting.")
    else:
        # 1. Show the analyzing spinner while the agent runs
        with st.spinner("Analyzing claim and matching policy rules..."):
            result = run_claim(user_input)
        
        # 2. Display the Decision Alert box
        decision = result["decision"].upper()
        if "APPROVE" in decision:
            st.success(f"**Decision:** {decision}")
        elif "DENY" in decision or "EXCLUDE" in decision:
            st.error(f"**Decision:** {decision}")
        else:
            st.warning(f"**Decision:** {decision}")
            
        # 3. Display the Reasoning
        st.markdown("### Reasoning:")
        st.write(result["reasoning"])
        
        # 4. Display the Retrieved Policy Chunks
        st.markdown("---")
        st.markdown("### Retrieved Policy Chunks")
        if result.get("chunks"):
            for i, chunk in enumerate(result["chunks"], 1):
                with st.expander(f"Reference Policy Chunk #{i}", expanded=True):
                    st.info(chunk)
        else:
            st.info("No relevant policy sections were matched.")
