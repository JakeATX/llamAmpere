"""Verify cache restore against the files and token prefix captured at save time."""
import hashlib
from pathlib import Path


def saved_prefix(saved, prompt_tokens, generated_tokens):
    count = saved.get('n_saved')
    if type(count) is not int or not prompt_tokens <= count <= prompt_tokens + generated_tokens:
        raise ValueError('cache did not save the expected prompt prefix')
    return count


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def verify_roundtrip(directory, original, restored, saved_count, original_prefix='prompt.bin', restored_prefix='roundtrip.bin'):
    directory = Path(directory)
    if type(saved_count) is not int or saved_count <= 0 or type(restored.get('n_restored')) is not int or restored['n_restored'] != saved_count:
        raise ValueError('restored cache token count differs from saved prefix')
    if not original or original_prefix not in original:
        raise ValueError('original main cache file is missing')
    expected = set()
    for name in original:
        if Path(name).name != name or not name.startswith(original_prefix):
            raise ValueError('invalid original cache file name')
        expected.add(restored_prefix + name[len(original_prefix):])
    if {p.name for p in directory.glob(restored_prefix + '*')} != expected:
        raise ValueError('round-trip cache sidecar set differs')
    if {p.name for p in directory.glob(original_prefix + '*')} != set(original):
        raise ValueError('original cache sidecar set changed')
    equal = {}
    for name, record in original.items():
        other = restored_prefix + name[len(original_prefix):]
        for candidate in (directory / name, directory / other):
            if candidate.is_symlink() or not candidate.is_file() or type(record['bytes']) is not int or record['bytes'] <= 0:
                raise ValueError('invalid or empty cache file')
            if candidate.stat().st_size != record['bytes'] or file_hash(candidate) != record['sha256']:
                raise ValueError('cache file differs from its original saved bytes: ' + candidate.name)
        equal[name] = True
    return equal
