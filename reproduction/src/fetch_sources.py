"""Archive cited primary paper and supplementary file with hashes."""
from common import *
import urllib.request,hashlib,datetime,zipfile,xml.etree.ElementTree as ET,traceback
def run():
    folder=REP/'sources';folder.mkdir(exist_ok=True)
    urls={'paper.html':'https://link.springer.com/article/10.1186/s42825-025-00215-8',
        'supplementary.docx':'https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs42825-025-00215-8/MediaObjects/42825_2025_215_MOESM1_ESM.docx'}
    records=[]
    for name,url in urls.items():
        try:
            blob=urllib.request.urlopen(url,timeout=90).read();(folder/name).write_bytes(blob)
            records.append(dict(file=name,url=url,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),status='downloaded'))
        except Exception:records.append(dict(file=name,url=url,status='failed',error=traceback.format_exc()))
    write_json(folder/'source_manifest.json',records)
    f=folder/'supplementary.docx'
    if f.exists():
        with zipfile.ZipFile(f) as z:
            root=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs=[''.join(p.itertext()) for p in root.findall('.//w:p',ns)]
            (folder/'supplementary_extracted.txt').write_text('\n'.join(paragraphs))
            tables=[]
            for table in root.findall('.//w:tbl',ns):
                tables.append([[''.join(c.itertext()) for c in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)])
            write_json(folder/'supplementary_tables.json',tables)
    print(records)
if __name__=='__main__':run()
