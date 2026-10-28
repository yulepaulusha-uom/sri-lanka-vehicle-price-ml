"""Prepare a separate reviewable repository directory without Git initialization."""
from pathlib import Path
import json,shutil,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',default='build/github');args=parser.parse_args()
out=(ROOT/args.output).resolve()
if out==ROOT or not out.is_relative_to(ROOT):raise ValueError('Output must be inside assignment.')
if out.exists():raise FileExistsError('Choose a new review directory; no silent replacement.')
manifest=json.loads((ROOT/'deployment/github-files.json').read_text())
if manifest['private_path_findings']:raise ValueError('Resolve privacy findings first.')
for rel in manifest['files']:
    dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
print('Prepared',len(manifest['files']),'files in separate review directory. No .git, commit, remote or push created.')
