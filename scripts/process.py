import subprocess, sys, os

# install requirements before actual code
req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

# --- rest of your processing logic ---
print("All dependencies installed!")
