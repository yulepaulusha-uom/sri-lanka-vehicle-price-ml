"""Assemble a directory for Azure code deployment; never includes raw data or training checkpoints."""
from pathlib import Path
import argparse,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]


def prepare(destination):
    destination=Path(destination).resolve()
    if not destination.is_relative_to(ROOT) or destination==ROOT:
        raise ValueError('Build directory must be inside this assignment.')
    if destination.exists():raise FileExistsError('Use a fresh build path; existing output is not overwritten.')
    destination.mkdir(parents=True)
    paths=['app.py','src/__init__.py','src/data.py','src/inference.py','src/deployment_runtime.py',
           'src/ebm_extension/__init__.py','src/ebm_extension/data.py','src/ebm_extension/inference.py',
           'models/accepted/pipeline.joblib','models/accepted/fit.json','results/ebm_extension/manifest.json',
           'models/ebm_extension/ebm-local-v1/model.joblib']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'results/accepted').rglob('*') if p.is_file()]
    base='results/ebm_extension/runs/ebm-local-v1/'
    paths += [base+p for p in ['evaluation.json','comparison.csv','category_counts.json','local_example.json']]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/base/'figures').glob('*.png')]
    for relative in sorted(set(paths)):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    shutil.copy2(ROOT/'deployment/requirements.txt',destination/'requirements.txt')
    (destination/'.streamlit').mkdir()
    shutil.copy2(ROOT/'deployment/streamlit-config.toml',destination/'.streamlit/config.toml')
    # Keep shell startup LF-terminated for Linux.
    (destination/'startup.sh').write_text((ROOT/'deployment/startup.sh').read_text(),encoding='utf8',newline='\n')
    entries=[dict(path=p.relative_to(destination).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(destination.rglob('*')) if p.is_file()]
    assert not (destination/'data/car_price_dataset.csv').exists()
    (destination/'runtime_manifest.json').write_text(json.dumps(dict(files=entries,total_bytes=sum(e['bytes'] for e in entries)),indent=2),encoding='utf8')
    print(f'Prepared {len(entries)} runtime files, {sum(e["bytes"] for e in entries)/2**20:.2f} MiB')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='build/azure');args=parser.parse_args()
    prepare(ROOT/args.output)
