"""Record reproducible setup inputs and check installed dependency health."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def fingerprint():
    digest = hashlib.sha256(sys.version.encode())
    for path in sorted([ROOT / 'requirements.txt', *[p for p in (ROOT / 'vendor').rglob('*')
                      if p.is_file() and ('src' in p.parts or p.name in ('pyproject.toml', 'setup.cfg'))
                      and '__pycache__' not in p.parts and p.suffix != '.pyc'
                      and not any(part.endswith('.egg-info') for part in p.parts)]]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

if __name__ == '__main__':
    if sys.argv[1] == 'fingerprint':
        print(fingerprint())
    else:
        import lightgbm, xgboost, thrember, huggingface_hub, matplotlib
        import shutil
        command = ([shutil.which('uv'), 'pip', 'check', '--python', sys.executable]
                   if shutil.which('uv') else [sys.executable, '-m', 'pip', 'check'])
        sys.exit(subprocess.run(command).returncode)
