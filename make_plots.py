import subprocess
from pathlib import Path
import sys

# Root path to the parent folder
base_dir = Path(r"C:\Users\reill\lab\nao_project\logs-robotics-dev\skrl\nao_flat")

# Get subfolders sorted alphabetically (oldest first)
subfolders = sorted([f for f in base_dir.iterdir() if f.is_dir()])

for i, folder in enumerate(subfolders, start=1):
    checkpoint_dir = folder / "checkpoints"
    if not checkpoint_dir.exists():
        continue

    # Get checkpoint files, sorted reverse alphabetical
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"), reverse=True)

    if not checkpoint_files:
        continue

    # Pick the reverse alphabetical first file (best_agent.pt if present)
    checkpoint = checkpoint_files[0]

    # Run the Isaac Lab command
    cmd = [
        r"..\..\Isaaclab\isaaclab.bat",
        "-p", "scripts\\skrl\\play-analyse.py",
        "--task", "Nao-Mgr-Play-Forward",
        "--num_envs", "1",
        "--headless",
        "--checkpoint", str(checkpoint),
        "--iteration", str(i)  # New argument you'll add to play-analyse.py
    ]
    print(f"Running iteration {i} with checkpoint: {checkpoint}")
    subprocess.run(cmd)
