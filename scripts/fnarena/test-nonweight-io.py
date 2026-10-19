import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonweight_io import activation_metadata

root='/mnt/models/fnarena/owned'
line=f'FNARENA_META_IO op=posix_fallocate start=11.1 end=11.2 tid=12 result=0 read_bytes=163840 path={root}/llama-moe-output-Abc123 (deleted)'
def run(log=line,start=11,end=12,total=163840):return activation_metadata(log,start,end,total,root)
assert run()['remaining_read_bytes']==0
assert run(total=163841)['remaining_read_bytes']==1
created=f'FNARENA_META_IO op=mkstemp start=11.3 end=11.4 tid=12 result=43 read_bytes=139264 path={root}/llama-moe-output-Def456'
assert run(created,total=139264)['remaining_read_bytes']==0
assert run(line+'\n'+created,total=303104)['remaining_read_bytes']==0
for bad in [created.replace('result=43','result=-1'),created.replace(root,'/mnt/models/other'),created+' (deleted)',created.replace('mkstemp','unlink'),created.replace('Def456','../foo')]:
 assert run(bad,total=139264)['activation_metadata_bytes']==0
for bad in [created+'\n'+created,created.replace('11.3','11.15').replace('11.4','11.25')+'\n'+line]:
 try:run(bad,total=1000000)
 except ValueError:pass
 else:raise AssertionError('creation overlap accepted')
for bad in [line.replace(root,'/mnt/models/other'),line.replace('posix_fallocate','pread'),line.replace('result=0','result=1'),line.replace('(deleted)',''),line.replace('Abc123','a/b123')]:
 assert run(bad)['activation_metadata_bytes']==0
for args in [dict(log=line+'\n'+line,total=327680),dict(total=163839),dict(start=11.15),dict(end=11.15),dict(total=None),dict(start=float('nan')),dict(end=10)]:
 try:run(**args)
 except ValueError:pass
 else:raise AssertionError(args)
print('Owned activation metadata, byte conservation, scope, boundary, duplicate and invalid-counter checks PASS')
