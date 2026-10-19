"""Account only traced metadata I/O for an owned activation temporary file."""
import math
import re
from pathlib import Path


def activation_copies(log, start, end, total, output_dir):
    """Count only pure memcpy reads from a verified owned activation-file mapping."""
    if type(total) is not int or total < 0 or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('invalid request I/O interval or counter')
    if 'FNARENA_ACTIVATION_IO_ERROR' in log:
        raise ValueError('activation copy counter unavailable')
    own_path = re.compile(re.escape(str(Path(output_dir).resolve())) + r'/llama-moe-output-[A-Za-z0-9]{6} \(deleted\)$')
    pattern = re.compile(r'^FNARENA_ACTIVATION_IO start=([0-9.]+) end=([0-9.]+) tid=([0-9]+) read_bytes=([0-9]+) offset=([0-9]+) bytes=([0-9]+) scope=owned_output_memcpy path=(.+)$')
    windows = {}
    observed = 0
    calls = 0
    for line in log.splitlines():
        if not line.startswith('FNARENA_ACTIVATION_IO '):
            continue
        match = pattern.fullmatch(line)
        if not match or not own_path.fullmatch(match[7]):
            raise ValueError('invalid activation copy source or trace')
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
                intervals=[(tid, begin, finish) for tid, spans in windows.items() for begin, finish in spans],
                scope='Only pure llama_copy_moe_output memcpy from the verified owned activation-file mapping, after context synchronization; no model-weight reads exempted.')


def activation_metadata(log, start, end, total, output_dir):
    if type(total) is not int or total < 0 or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('invalid request I/O interval or counter')
    own_path = re.escape(str(Path(output_dir).resolve())) + r'/llama-moe-output-[A-Za-z0-9]{6}'
    pattern = re.compile(r'^FNARENA_META_IO op=(posix_fallocate|mkstemp) start=([0-9.]+) end=([0-9.]+) tid=([0-9]+) result=(-?[0-9]+) read_bytes=([0-9]+) path=(.+)$')
    windows = {}
    observed = 0
    for line in log.splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        op, result = match[1], int(match[5])
        # mkstemp creates a new empty file with O_EXCL. It cannot read a model
        # tensor. Allocation is scoped separately to the subsequently unlinked file.
        suffix = r' \(deleted\)' if op == 'posix_fallocate' else ''
        if not re.fullmatch(own_path + suffix, match[7]) or (result != 0 if op == 'posix_fallocate' else result < 0):
            continue
        begin, finish = float(match[2]), float(match[3])
        if not math.isfinite(begin) or not math.isfinite(finish) or finish < begin:
            raise ValueError('invalid metadata trace interval')
        if finish < start or begin > end:
            continue
        if not start <= begin <= finish <= end:
            raise ValueError('metadata interval crosses request boundary')
        tid, count = int(match[4]), int(match[6])
        if tid <= 0 or count <= 0:
            raise ValueError('invalid metadata thread or byte count')
        if any(begin <= previous_end and finish >= previous_begin for previous_begin, previous_end in windows.get(tid, [])):
            raise ValueError('duplicate or overlapping metadata attribution')
        windows.setdefault(tid, []).append((begin, finish))
        observed += count
    if observed > total:
        raise ValueError('metadata attribution exceeds process reads')
    return dict(activation_metadata_bytes=observed, remaining_read_bytes=total-observed,
                intervals=[(tid, begin, finish) for tid, spans in windows.items() for begin, finish in spans],
                scope='Only successful, fully contained creation of new empty owned activation files and allocation of owned deleted activation files; no model/activation payload reads exempted.')


def activation_device_copies(log, start, end, total, output_dir):
    """Account exact device-to-owned-output operations from the verified CUDA probe."""
    from bisect import bisect_left
    if type(total) is not int or total < 0 or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('invalid device output interval or counter')
    if 'FNSHIP_NONWEIGHT_PROBE_ERROR' in log:
        raise ValueError('device output counter unavailable')
    own_path = re.compile(re.escape(str(Path(output_dir).resolve())) + r'/llama-moe-output-[A-Za-z0-9]{6} \(deleted\)$')
    pattern = re.compile(r'^FNSHIP_NONWEIGHT_PROBE op=(cuda_d2h_owned|cuda_d2h_async_owned|cuda_d2h_2d_owned|cuda_d2h_2d_async_owned) start=([0-9.]+) end=([0-9.]+) tid=([0-9]+) result=0 read_bytes=([0-9]+) path=(.+)$')
    observed = 0
    windows = {}
    calls = 0
    marker = 'FNSHIP_NONWEIGHT_PROBE '
    if log and not log.endswith('\n'):
        tail = log.rsplit('\n', 1)[-1]
        if tail and (marker in tail or marker.startswith(tail)):
            raise ValueError('incomplete device output trace')
    for line in log.splitlines():
        if not line.startswith(marker):
            continue
        operation = line[len(marker):].split(' ', 1)[0]
        if operation not in ('op=cuda_d2h_owned', 'op=cuda_d2h_async_owned', 'op=cuda_d2h_2d_owned', 'op=cuda_d2h_2d_async_owned'):
            continue
        match = pattern.fullmatch(line)
        if not match or not own_path.fullmatch(match[6]):
            raise ValueError('invalid device output source or trace')
        begin, finish = float(match[2]), float(match[3])
        if not math.isfinite(begin) or not math.isfinite(finish) or finish < begin:
            raise ValueError('invalid device output trace interval')
        if finish < start or begin > end:
            continue
        if not start <= begin <= finish <= end:
            raise ValueError('device output trace crosses request boundary')
        tid, count = int(match[4]), int(match[5])
        if tid <= 0 or count <= 0:
            raise ValueError('invalid device output thread or count')
        spans = windows.setdefault(tid, [])
        index = bisect_left(spans, (begin,))
        if ((index and begin <= spans[index-1][1]) or (index < len(spans) and finish >= spans[index][0])):
            raise ValueError('duplicate or overlapping device output attribution')
        spans.insert(index, (begin, finish))
        observed += count
        calls += 1
    if observed > total:
        raise ValueError('device output attribution exceeds process reads')
    return dict(activation_device_copy_bytes=observed, positive_calls=calls, remaining_read_bytes=total-observed,
                intervals=[(tid, begin, finish) for tid, spans in windows.items() for begin, finish in spans],
                scope='Successful CUDA device-to-owned-output copies only; exact device source checked before I/O counters, no context synchronization or host callback.')
