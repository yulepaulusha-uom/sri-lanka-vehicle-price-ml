"""Execute the academic notebook using this isolated environment without kernel installation."""
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['IPYTHONDIR']=str(Path.cwd()/'.ipython-ebm')
os.environ['JUPYTER_PATH']=str(Path.cwd()/'.jupyter-ebm')
os.environ['MPLCONFIGDIR']=str(Path.cwd()/'.matplotlib-ebm')
import json,sys
import nbformat
from nbclient import NotebookClient
root=Path.cwd()
spec=root/'.jupyter-ebm/kernels/ebm'
spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Python EBM isolated',language='python')))
path=root/'vehicle_price_ebm_extension.ipynb'
notebook=nbformat.read(path,as_version=4)
client=NotebookClient(notebook,timeout=7200,kernel_name='ebm',resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    run_id=os.environ.get('EBM_RUN_ID','ebm-local-v1')
    if run_id!='ebm-local-v1':
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Unsafe EBM_RUN_ID')
        output=root/'results/ebm_extension/runs'/run_id/'executed_notebook.ipynb'
        output.parent.mkdir(parents=True,exist_ok=True)
    else: output=path
    nbformat.write(notebook,output)
print('Notebook executed and saved:',output)
