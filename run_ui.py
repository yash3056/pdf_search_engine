"""
Direct Streamlit UI runner.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Streamlit default port is 8501
    ui_port = int(os.getenv("STREAMLIT_PORT", 8501))
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/ui.py",
        "--server.port", str(ui_port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    subprocess.run(cmd)
