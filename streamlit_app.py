# Entry point for Streamlit Community Cloud
import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the app
import src.ui.app