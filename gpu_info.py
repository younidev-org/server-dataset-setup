"""Report GPU visibility and the precise XGBoost CUDA probe result."""
import json
import subprocess
import xgboost as xgb
from experiment import resolve_xgboost_device

result = {'xgboost_version': xgb.__version__, 'build': xgb.build_info()}
try:
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    result['nvidia_smi'] = {'returncode': query.returncode, 'stdout': query.stdout.strip(),
                            'stderr': query.stderr.strip()}
except (OSError, subprocess.TimeoutExpired) as exc:
    result['nvidia_smi'] = {'error': str(exc)}
result['cuda_probe'] = resolve_xgboost_device('auto', 1)
print(json.dumps(result, indent=2))
