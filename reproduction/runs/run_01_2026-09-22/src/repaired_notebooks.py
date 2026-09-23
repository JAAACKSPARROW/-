"""Real notebook execution with path-only repairs; author source stays intact."""
from common import *
import traceback,copy,difflib
def run():
    import nbformat,pandas as pd
    from nbclient import NotebookClient
    runtime=REP/'logs/jupyter_runtime';runtime.mkdir(parents=True,exist_ok=True)
    (REP/'cache/ipython').mkdir(parents=True,exist_ok=True)
    os.environ['IPYTHONDIR']=str(REP/'cache/ipython');os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    work=RESULT/'original_notebook_outputs';work.mkdir(exist_ok=True)
    statuses=[]
    for f in sorted((AUTHOR/'Model_development').glob('*.ipynb')):
        nb=nbformat.read(f,as_version=4);patches=[]
        for i,c in enumerate(nb.cells):
            if c.cell_type!='code':continue
            old=c.source;c.outputs=[];c.execution_count=None
            if 'ESM_embedding' in f.name:
                if i==0:c.source=f'import os\nos.chdir({str(work)!r})'
                if i==3:c.source=f"import pandas as pd\ndataset=pd.read_excel({str(AUTHOR/'Database/Oringinal_data.xlsx')!r}, na_filter=False)\nsequence_list=dataset['Sequence']"
            else:
                if i==0:c.source=c.source.replace('embedding_results/',str(AUTHOR/'Database')+'/')
                if i==2:c.source='import sys\nsys.path.insert(0,'+repr(str(AUTHOR))+')\n'+c.source
            if old!=c.source:patches.extend(difflib.unified_diff(old.splitlines(True),c.source.splitlines(True),fromfile=f'{f.name} cell {i} original',tofile='path repaired'))
        status='PASS';err=''
        try:NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(work)}}).execute()
        except Exception as e:
            err=str(e);status='PARTIAL' if any(c.get('execution_count') for c in nb.cells) else 'FAIL'
            (REP/f'logs/{f.stem}_path_repaired_error.log').write_text(traceback.format_exc())
        nbformat.write(nb,REP/'executed_notebooks'/('path_repaired_'+f.name))
        (REP/f'logs/{f.stem}_path_repair.diff').write_text('\n'.join(patches))
        statuses.append(dict(notebook=f.name,status=status,error=err,modifications='paths/import resolution only'))
        pd.DataFrame(statuses).to_csv(RESULT/'path_repaired_notebook_status.csv',index=False)
        print(f.name,status,flush=True)
if __name__=='__main__':run()
