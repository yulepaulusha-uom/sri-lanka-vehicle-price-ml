"""Distinguish immutable historical evidence from authorized document revisions."""
import json
from src.data import ROOT,sha256


def check_assignment_integrity():
    before=json.loads((ROOT/'verification/ebm_extension/integrity_before.json').read_text())
    revisions=set(before['authorized_revision_backups'])
    # A curated repository declares machine-specific archival files it omits.
    repo_manifest=ROOT/'deployment/github-files.json'
    repo_files=set(json.loads(repo_manifest.read_text())['files']) if repo_manifest.exists() else None
    checked=[];unavailable=[]
    for name,digest in before['protected_files'].items():
        # The initial broad ledger also captured the authorized inventory index.
        # Its original bytes remain in the historical copy; updating it is allowed.
        if name in revisions:
            assert sha256(ROOT/'verification/ebm_extension/historical_copies'/name)==digest,name
            continue
        optional=(name.startswith('results/runs/main-local-20261020/') or
                  name.startswith('verification/partial_promotion/') or name=='.venv/pyvenv.cfg')
        path=ROOT/name
        if not path.exists() and (optional or (repo_files is not None and name not in repo_files)):
            unavailable.append(name)
            continue
        assert path.is_file(),name
        assert sha256(path)==digest,name
        checked.append(name)
    for name,digest in before['authorized_revision_backups'].items():
        assert sha256(ROOT/'verification/ebm_extension/historical_copies'/name)==digest,name
    return dict(immutable_checked=len(checked),optional_local_evidence_unavailable=unavailable,
                authorized_inventory_revision_preserved_in_backup=True)
