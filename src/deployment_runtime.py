"""Portable runtime loading; accepted EBM research sources/artifacts stay unchanged."""
import json
from pathlib import PurePosixPath
import joblib
from src.data import ROOT,sha256


def portable_relative(value):
    relative=PurePosixPath(value.replace('\\','/'))
    if relative.is_absolute() or '..' in relative.parts or ':' in relative.parts[0]:
        raise ValueError('Expected an application-relative artifact path.')
    return relative.as_posix()


def load_ebm():
    manifest=json.loads((ROOT/'results/ebm_extension/manifest.json').read_text())
    normalized=dict(manifest)
    for key in ['model','results']:
        normalized[key]=portable_relative(manifest[key])
    path=ROOT/normalized['model']
    if sha256(path)!=manifest['sha256']:
        raise ValueError('EBM model checksum differs.')
    return joblib.load(path),normalized
