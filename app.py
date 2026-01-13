import streamlit as st
import requests

st.set_page_config(page_title="File Downloader", layout="centered")

st.title("📥 Download File")

github_raw_url = st.text_input(
    "Enter GitHub RAW Python file URL",
    "https://raw.githubusercontent.com/KrishnaSarovar/ADD/main/wt.html"
)

if st.button("Fetch File"):
    try:
        response = requests.get(github_raw_url)
        response.raise_for_status()

        st.success("File fetched successfully!")

        st.code(response.text, language="python")

        st.download_button(
            label="Download",
            data=response.text,
            file_name="wt.html",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error fetching file: {e}")


