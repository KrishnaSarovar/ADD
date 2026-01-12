import streamlit as st
import requests

st.set_page_config(page_title="GitHub File Downloader", layout="centered")

st.title("📥 Download Python File from GitHub")

github_raw_url = st.text_input(
    "Enter GitHub RAW Python file URL",
    "https://raw.githubusercontent.com/KrishnaSarovar/ADD/main/DAA.py"
)

if st.button("Fetch File"):
    try:
        response = requests.get(github_raw_url)
        response.raise_for_status()

        st.success("File fetched successfully!")

        st.code(response.text, language="python")

        st.download_button(
            label="Download DAA.py",
            data=response.text,
            file_name="DAA.py",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error fetching file: {e}")
