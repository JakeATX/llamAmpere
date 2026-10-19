"""Reject unproved operations, overlap, cross-request reads and incomplete records."""
from nonweight_io import activation_device_copies
root='/test'
line='FNSHIP_NONWEIGHT_PROBE op=cuda_d2h_async_owned start=2.0 end=3.0 tid=7 result=0 read_bytes=4096 path=/test/llama-moe-output-Ab1234 (deleted)\n'
def run(text=line,start=1,end=4,total=8192):return activation_device_copies(text,start,end,total,root)
assert run()['activation_device_copy_bytes']==4096
assert run()['remaining_read_bytes']==4096
for op in ['cuda_d2h_owned','cuda_d2h_2d_owned','cuda_d2h_2d_async_owned']:assert run(line.replace('cuda_d2h_async_owned',op))['activation_device_copy_bytes']==4096
for op in ['cuda_h2d_owned','cuda_h2d_async_owned','msync_owned','close_owned','unlink_empty','pread']:assert run(line.replace('cuda_d2h_async_owned',op))['activation_device_copy_bytes']==0
assert run(start=3.1,end=4)['activation_device_copy_bytes']==0
assert run(line+line.replace('tid=7','tid=8'))['activation_device_copy_bytes']==8192
for bad in [line+line,line+line.replace('2.0','2.5'),line.replace('/test/','/weights/'),line.replace(' (deleted)',''),line.replace('Ab1234','../bad'),line.replace('result=0','result=1'),line.replace('read_bytes=4096','read_bytes=0'),line.replace('tid=7','tid=0'),line.replace('start=2.0','start=nan'),'FNSHIP_NONWEIGHT_PROBE_ERROR op=cuda_d2h_owned\n']:
 try:run(bad)
 except ValueError:pass
 else:raise AssertionError(bad)
for n in range(1,len(line)):
 try:run(line[:n])
 except ValueError:pass
 else:
  if n<len('FNSHIP_NONWEIGHT_PROBE ') or line[:n].startswith('FNSHIP_NONWEIGHT_PROBE '):raise AssertionError(('partial',n))
for kw in [dict(total=4095),dict(total=None),dict(total=True),dict(start=2.1),dict(end=2.9),dict(start=float('nan')),dict(end=0)]:
 try:run(**kw)
 except ValueError:pass
 else:raise AssertionError(kw)
print('PASS device output scope, four operations, byte conservation, thread overlap, boundaries, malformed and all partial prefixes')
