"""Cross-platform environment setup: python setup_project.py [--ai]."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai', action='store_true', help='Install MPNet/BART dependencies as well')
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error('Use Python 3.10, 3.11 or 3.12 (3.11 recommended) for compatible AI wheels.')
    root = Path(__file__).resolve().parent
    env = root / '.venv'
    if not env.exists():
        venv.create(env, with_pip=True)
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    requirement = 'requirements-ai.txt' if args.ai else 'requirements.txt'
    subprocess.check_call([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], cwd=root)
    if args.ai and sys.platform != 'darwin':
        # CPU wheels keep the local setup independent of CUDA toolkits.
        subprocess.check_call([str(python), '-m', 'pip', 'install', 'torch>=2.6,<3', '--index-url', 'https://download.pytorch.org/whl/cpu'], cwd=root)
    subprocess.check_call([str(python), '-m', 'pip', 'install', '-r', requirement], cwd=root)
    print('\nSetup complete. In VS Code, select the .venv interpreter and press F5.')
    print(f'Alternatively run: "{python}" run.py')
    print('Then open http://127.0.0.1:8000')
