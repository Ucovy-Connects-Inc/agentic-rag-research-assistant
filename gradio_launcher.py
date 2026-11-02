"""
Simple launcher for the Gradio interface.
Run this script to start the web UI for the arXiv Paper Curator RAG system.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.gradio_app import main

if __name__ == "__main__":
    main()

