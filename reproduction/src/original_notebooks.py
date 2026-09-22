"""Execute unmodified notebooks in a disposable workspace-owned working directory."""
from common import *
import traceback, shutil
def run():
    import nbformat,pandas as pd
    from nbclient import NotebookClient
    work=REP/'executed_notebooks/original_workdir';work.mkdir(exist_ok=True)
    runtime=REP/'logs/jupyter_runtime';runtime.mkdir(exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    os.environ['IPYTHONDIR']=str(REP/'cache/ipython')
    outcomes=[]
    for file in sorted((AUTHOR/'Model_development').glob('*.ipynb')):
        nb=nbformat.read(file,as_version=4)
        for c in nb.cells:
            if c.cell_type=='code':c.outputs=[];c.execution_count=None
        status='PASS';error=''
        try:NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(work)}}).execute()
        except Exception as e:
            status='FAIL';error=str(e)
            with (REP/'logs/original_code_errors.log').open('a') as f:f.write(f'\n{file.name}\n{traceback.format_exc()}\n')
        nbformat.write(nb,REP/'executed_notebooks'/file.name)
        outcomes.append(dict(notebook=file.name,status=status,error=error,source_changed=False))
        print(file.name,status,flush=True)
    pd.DataFrame(outcomes).to_csv(RESULT/'original_notebook_status.csv',index=False)
if __name__=='__main__':run()
