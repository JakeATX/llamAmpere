"""Copy the tested build and its project DSOs into an immutable test bundle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--name-only'], text=True).strip()
source = root / 'build/bin'
destination = root / 'releases' / commit[:9] / 'bin'
destination.mkdir(parents=True, exist_ok=False)
targets = ('llama-server', 'llama-cli', 'test-moe-arena', 'test-moe-cache')
for path in source.iterdir():
    if '.so' not in path.name and path.name not in targets:
        continue
    if path.is_symlink():
        (destination / path.name).symlink_to(os.readlink(path))
    else:
        shutil.copy2(path, destination / path.name)
for name in targets:
    (destination / ('llama.cpp-' + name.removeprefix('llama-'))).symlink_to(name)
environment = dict(os.environ, LD_LIBRARY_PATH=str(destination))
dependencies = subprocess.check_output(['ldd', str(destination / 'llama.cpp-server')], env=environment, text=True)
assert 'not found' not in dependencies
for line in dependencies.splitlines():
    if any(name in line for name in ('libllama', 'libggml', 'libmtmd')):
        assert str(destination) in line, line
manifest = {'source_commit': commit, 'libraries': {}, 'dependencies': dependencies}
for path in destination.iterdir():
    if path.is_symlink():
        continue
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert digest == hashlib.file_digest((source / path.name).open('rb'), 'sha256').hexdigest()
    manifest['libraries'][path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
(destination.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(destination.parent)
