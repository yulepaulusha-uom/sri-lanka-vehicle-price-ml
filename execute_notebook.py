"""Execute the academic notebook with this project's own Python kernel."""
import os
os.environ["MPLCONFIGDIR"] = str(__import__('pathlib').Path(__file__).resolve().parent / ".matplotlib")
os.environ["IPYTHONDIR"] = str(__import__('pathlib').Path(__file__).resolve().parent / ".ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(__import__('pathlib').Path(__file__).resolve().parent / ".jupyter/runtime")
import json
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parent
notebook=root/"vehicle_price_assignment.ipynb"
nb=nbformat.read(notebook,as_version=4)
nbformat.validate(nb)
km=KernelManager()
km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
client=NotebookClient(nb,timeout=7200,km=km,resources={"metadata":{"path":str(root)}})
client.on_cell_start = lambda cell, cell_index: print(f"Executing notebook cell {cell_index}: {cell.cell_type}", flush=True)
try:
    client.execute()
finally:
    # Preserve partial outputs if an execution is interrupted or raises an error.
    nbformat.write(nb,notebook)
    if km.has_kernel:
        km.shutdown_kernel(now=True)
print("Executed notebook cells:",sum(c.cell_type=="code" for c in nb.cells))
