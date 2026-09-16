"""
Start the whole system with one command.

    python run.py                 # train if needed, start the backend and the dashboard
    python run.py --no-browser    # same, without opening a browser tab

Ctrl+C stops everything. If a backend or a dashboard is already running on its
port it is reused instead of failing with "address already in use".
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import BACKEND_HOST, BACKEND_PORT, BACKEND_URL, MODEL_PATH  # noqa: E402

DASHBOARD_PORT = 8501
DASHBOARD_URL = f"http://localhost:{DASHBOARD_PORT}"


def alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:                                 # noqa: BLE001 - any failure means "not up"
        return False


def wait_for(url: str, seconds: int, label: str) -> bool:
    for _ in range(seconds):
        if alive(url):
            return True
        time.sleep(1)
    print(f"  {label} did not answer within {seconds} s")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-time prediction system")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("No trained model found - downloading data and training first (about a minute)")
        if subprocess.run([sys.executable, str(ROOT / "train_model.py")], cwd=ROOT).returncode != 0:
            print("Training failed, see the messages above.")
            return 1

    children: List[subprocess.Popen] = []

    if alive(f"{BACKEND_URL}/health"):
        print(f"[1/2] Backend already running at {BACKEND_URL} - reusing it")
    else:
        print(f"[1/2] Starting the backend at {BACKEND_URL}")
        children.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)], cwd=ROOT))
        if not wait_for(f"{BACKEND_URL}/health", 90, "backend"):
            for child in children:
                child.terminate()
            return 1

    if alive(DASHBOARD_URL):
        print(f"[2/2] Dashboard already running at {DASHBOARD_URL} - reusing it")
    else:
        print(f"[2/2] Starting the dashboard at {DASHBOARD_URL}")
        children.append(subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
             "--server.port", str(DASHBOARD_PORT)], cwd=ROOT))
        wait_for(DASHBOARD_URL, 60, "dashboard")

    if not args.no_browser:
        webbrowser.open(DASHBOARD_URL)

    print()
    print("=" * 60)
    print(f"  Dashboard : {DASHBOARD_URL}")
    print(f"  API docs  : {BACKEND_URL}/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while children:
            time.sleep(1)
            for child in children:
                if child.poll() is not None:
                    print(f"A service exited with code {child.returncode}; shutting down.")
                    raise KeyboardInterrupt
        while True:                                   # nothing to babysit, just keep the console
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
