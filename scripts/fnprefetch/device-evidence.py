import re

def device_evidence(log):
    states = re.findall(r'^.*\[moe-cache\] (device tables (?:active|skipped|unavailable)[^\n]*|device preparation failed[^\n]*)', log, re.M)
    assert states and states[-1].startswith('device tables active: 144 tensors,'), 'device tables are not active for all target projections'
    counts = [int(x) for x in re.findall(r'\[moe-cache\] device dispatch: calls=(\d+)', log)]
    counts += [int(x) for x in re.findall(r'\[moe-cache\] device routes: hits=\d+ misses=\d+ dispatches=(\d+)', log)]
    assert counts and max(counts)>0, 'device flag was set but no device dispatch completed'
    return {'last_table_state':states[-1], 'maximum_logged_dispatches':max(counts)}
