import tempfile
from pathlib import Path
from cache_roundtrip import saved_prefix, verify_roundtrip, file_hash

assert saved_prefix({'n_saved': 8192},8192,128)==8192
assert saved_prefix({'n_saved': 8319},8192,128)==8319
for count in [0,1,8191,8321,None,True,'8192']:
    try:saved_prefix({'n_saved': count},8192,128)
    except ValueError:pass
    else:raise AssertionError(count)
with tempfile.TemporaryDirectory() as root:
    p=Path(root)
    originals={'prompt.bin': b'main-state','prompt.bin.mtp': b'helper-state'}
    for name,data in originals.items():
        (p/name).write_bytes(data)
        (p/name.replace('prompt.bin','roundtrip.bin',1)).write_bytes(data)
    records={name:dict(bytes=len(data),sha256=file_hash(p/name)) for name,data in originals.items()}
    def run(count=8192):return verify_roundtrip(p,records,{'n_restored':count},8192)
    assert all(run().values())
    for count in [0,8191,8193,None,True,'8192']:
        try:run(count)
        except ValueError:pass
        else:raise AssertionError(count)
    (p/'roundtrip.bin.extra').write_bytes(b'extra')
    try:run()
    except ValueError:pass
    else:raise AssertionError('extra sidecar accepted')
    (p/'roundtrip.bin.extra').unlink()
    # The old reread-vs-reread comparison falsely accepts this equal corruption.
    (p/'prompt.bin').write_bytes(b'corruption')
    (p/'roundtrip.bin').write_bytes(b'corruption')
    assert (p/'prompt.bin').read_bytes()==(p/'roundtrip.bin').read_bytes()
    try:run()
    except ValueError:pass
    else:raise AssertionError('both mutated originals accepted')
    (p/'prompt.bin').write_bytes(originals['prompt.bin']);(p/'roundtrip.bin').write_bytes(originals['prompt.bin'])
    (p/'roundtrip.bin.mtp').unlink()
    try:run()
    except ValueError:pass
    else:raise AssertionError('missing sidecar accepted')
    (p/'roundtrip.bin.mtp').symlink_to(p/'prompt.bin.mtp')
    try:run()
    except ValueError:pass
    else:raise AssertionError('symlink accepted')
print('PASS nonzero full prefix, restore count, saved-hash immutability, exact sidecars and regular files')
