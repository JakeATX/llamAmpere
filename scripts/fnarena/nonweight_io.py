"""Account only traced metadata I/O for an owned activation temporary file."""
import math
import re
from pathlib import Path


def activation_copies(log, start, end, total):
    """Count a verified staging API's thread-local activation refault reads only."""
    if type(total) is not int or total < 0 or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('invalid request I/O interval or counter')
    if 'FNARENA_ACTIVATION_IO_ERROR' in log:
        raise ValueError('activation copy counter unavailable')
    pattern = re.compile(r'^FNARENA_ACTIVATION_IO start=([0-9.]+) end=([0-9.]+) tid=([0-9]+) result=1 read_bytes=([0-9]+) first=([0-9]+) count=([0-9]+) scope=nextn_output_copy$')
    windows = {}
    observed = 0
    calls = 0
    for line in log.splitlines():
        if not line.startswith('FNARENA_ACTIVATION_IO '):
            continue
        match = pattern.fullmatch(line)
        if not match:
            raise ValueError('invalid or failed activation copy trace')
        begin, finish = float(match[1]), float(match[2])
        if not math.isfinite(begin) or not math.isfinite(finish) or finish < begin:
            raise ValueError('invalid activation copy interval')
        if finish < start or begin > end:
            continue
        if not start <= begin <= finish <= end:
            raise ValueError('activation copy interval crosses request boundary')
        tid, count, rows = int(match[3]), int(match[4]), int(match[6])
        if tid <= 0 or count <= 0 or rows <= 0:
            raise ValueError('invalid activation copy count')
        if any(begin <= b and finish >= a for a, b in windows.get(tid, [])):
            raise ValueError('duplicate or overlapping activation copy attribution')
        windows.setdefault(tid, []).append((begin, finish))
        observed += count
        calls += 1
    if observed > total:
        raise ValueError('activation attribution exceeds process reads')
    return dict(activation_copy_bytes=observed, positive_calls=calls, remaining_read_bytes=total-observed,
                scope='Only successful, fully contained llama_copy_embeddings_nextn activation staging calls in the verified runtime; no model-weight reads exempted.')


def activation_metadata(log, start, end, total, output_dir):
    if type(total) is not int or total < 0 or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('invalid request I/O interval or counter')
    own_path = re.compile(re.escape(str(Path(output_dir).resolve())) + r'/llama-moe-output-[A-Za-z0-9]{6} \(deleted\)$')
    pattern = re.compile(r'^FNARENA_META_IO op=posix_fallocate start=([0-9.]+) end=([0-9.]+) tid=([0-9]+) result=0 read_bytes=([0-9]+) path=(.+)$')
    windows = {}
    observed = 0
    for line in log.splitlines():
        match = pattern.fullmatch(line)
        if not match or not own_path.fullmatch(match[5]):
            continue
        begin, finish = float(match[1]), float(match[2])
        if not math.isfinite(begin) or not math.isfinite(finish) or finish < begin:
            raise ValueError('invalid metadata trace interval')
        if finish < start or begin > end:
            continue
        if not start <= begin <= finish <= end:
            raise ValueError('metadata interval crosses request boundary')
        tid, count = int(match[3]), int(match[4])
        if tid <= 0 or count <= 0:
            raise ValueError('invalid metadata thread or byte count')
        if any(begin <= previous_end and finish >= previous_begin for previous_begin, previous_end in windows.get(tid, [])):
            raise ValueError('duplicate or overlapping metadata attribution')
        windows.setdefault(tid, []).append((begin, finish))
        observed += count
    if observed > total:
        raise ValueError('metadata attribution exceeds process reads')
    return dict(activation_metadata_bytes=observed, remaining_read_bytes=total-observed,
                scope='Only successful, fully contained posix_fallocate calls for this run owned deleted activation files; no model/activation payload reads exempted.')
