import hashlib, json, shutil, subprocess
from pathlib import Path
root=Path('/mnt/models/fnprefetch')
head=(root/'logs/build-source-head.txt').read_text().strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==head
assert not subprocess.check_output(['git','diff','--binary'],cwd=root)
source=root/'build/bin';target=root/'releases'/head[:9]/'bin'
target.mkdir(parents=True,exist_ok=False)
paths=[*source.glob('lib*.so*'),*[source/name for name in ('llama-server','test-moe-cache','test-backend-ops','test-moe-prefetch-model')]]
for path in paths:
 if path.is_symlink():
  (target/path.name).symlink_to(path.readlink())
 else:
  shutil.copy2(path,target/path.name)
(target/'llama.cpp-server').symlink_to('llama-server')
manifest={'commit':head,'files':{}}
for path in sorted(target.iterdir()):
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 manifest['files'][path.name]={'sha256':digest,'bytes':path.stat().st_size}
(target.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
pointer=root/'logs/frozen-bin.next'
pointer.write_text(str(target)+'\n');pointer.replace(root/'logs/frozen-bin.txt')
link=root/'llama.cpp-server'
next_link=root/'llama.cpp-server.next'
next_link.unlink(missing_ok=True);next_link.symlink_to(target/'llama-server');next_link.replace(link)
print('Frozen build:',target,flush=True)
