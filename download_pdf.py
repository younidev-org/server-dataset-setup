"""Download PDF train/test features and the shared EMBER2024 challenge set."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'datasets')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'data' / 'reports')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    downloads = []
    for filename, subdir in [('PDF_train.zip', 'PDF'), ('PDF_test.zip', 'PDF'),
                             ('challenge.zip', 'challenge')]:
        destination = data / subdir
        destination.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {filename} into {destination}', flush=True)
        archive = hf_hub_download(repo_id='joyce8/EMBER2024', filename=filename, repo_type='dataset')
        with zipfile.ZipFile(archive) as bundle:
            # Validate every member before extracting; never write outside its split.
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f'Unsafe archive member: {member.filename}')
            bundle.extractall(destination)
            files = [item for item in bundle.infolist() if not item.is_dir()]
            downloads.append({
                'archive': filename, 'destination': str(destination),
                'files': [str((destination / item.filename).relative_to(data)) for item in files],
                'extracted_bytes': sum(item.file_size for item in files),
            })
        # Keep the Hugging Face cache intact for retries and later runs.
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'repository': 'joyce8/EMBER2024', 'data_dir': str(data),
        'challenge_scope': 'Shared upstream challenge set; not restricted to PDF.',
        'downloads': downloads,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / 'download.json'
    temporary = report_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(report_path)
    print(f'Download and extraction complete. Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
