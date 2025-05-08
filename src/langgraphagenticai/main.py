from langgraphagenticai.ui.app import *

if __name__ == "__main__":
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run("src/langgraphagenticai/ui/streamlit_app.py", "", [], {})
