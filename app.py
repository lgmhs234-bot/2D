import streamlit as st
import pandas as pd
from agent import ask_sector_analyst

st.set_page_config(page_title="2D Animation Sector Intelligence", layout="wide")

st.title("🎨 2D Animation Sector Intelligence Engine")
st.markdown("AI-powered analytics and strategy assistant for **Vichar Desk**.")

api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

try:
    df = pd.read_csv("cleaned_2d_sector_dataset.csv")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reels Analyzed", len(df))
    col2.metric("Creators Tracked", df["ownerUsername"].nunique())
    col3.metric("Sector Median Views", f"{int(df['views'].median()):,}")
    col4.metric("Vichar Desk Median Views", f"{int(df[df['ownerUsername']=='vichardesk']['views'].median()):,}")
    st.markdown("---")
    
except Exception as e:
    st.error("Dataset `cleaned_2d_sector_dataset.csv` not found in repository.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about hook patterns, discussion rates, outlier formulas, or Vichar Desk strategy..."):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar first!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Analyzing 2D sector metrics and running code..."):
        try:
            answer = ask_sector_analyst(prompt, api_key)
        except Exception as e:
            answer = f"Error generating response: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
