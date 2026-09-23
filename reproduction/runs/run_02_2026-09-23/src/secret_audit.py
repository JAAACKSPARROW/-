"""Conservative staged-file secret scan; findings never print secret values."""
from common import *
import re,subprocess
def run():
    git='/Users/jingyuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git'
    if not Path(git).exists():git='git'
    entries=subprocess.check_output([git,'ls-files','--stage','-z'],cwd=ROOT).decode().split('\0')
    patterns={
        'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
        'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
        'openai_key':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
        'aws_access_key':re.compile(r'\bAKIA[A-Z0-9]{16}\b')}
    findings=[];inventory=[]
    for entry in entries:
        if not entry:continue
        meta,path=entry.split('\t',1)
        if meta.startswith('160000'):continue
        p=ROOT/path;size=p.stat().st_size;inventory.append(dict(path=path,size_bytes=size))
        if p.name=='.env' or p.name.startswith('.env.') or p.suffix in {'.pem','.key'} or p.name.lower() in {'credentials','id_rsa','id_ed25519'}:
            findings.append(dict(path=path,reason='sensitive filename'))
        if size>50*1024*1024:findings.append(dict(path=path,reason='unexpected >50MB staged file'))
        if p.suffix in {'.joblib','.npy','.keras','.png','.docx'}:continue
        content=p.read_text(errors='replace')
        for reason,pattern in patterns.items():
            for match in pattern.finditer(content):findings.append(dict(path=path,reason=reason,line=content.count('\n',0,match.start())+1))
    # Scan author tree as well: the parent gitlink does not embed its contents anew.
    for p in AUTHOR.rglob('*'):
        if not p.is_file() or '.git' in p.relative_to(AUTHOR).parts:continue
        if p.suffix not in {'.py','.md','.txt','.ipynb','.json','.yaml','.yml'}:continue
        content=p.read_text(errors='replace')
        for reason,pattern in patterns.items():
            if pattern.search(content):findings.append(dict(path=str(p.relative_to(ROOT)),reason=reason))
    write_json(REP/'logs/secret_audit.json',dict(status='PASS' if not findings else 'REVIEW_REQUIRED',staged_file_count=len(inventory),
        total_bytes=sum(x['size_bytes'] for x in inventory),largest_files=sorted(inventory,key=lambda x:x['size_bytes'],reverse=True)[:12],findings=findings,
        caveat='Basic filename/key-pattern scan, not proof that all possible secrets are absent. Keyword review performed separately.'))
    print('PASS' if not findings else findings)
    if findings:raise SystemExit(1)
if __name__=='__main__':run()
