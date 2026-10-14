#!/bin/bash
# Appends the AI+ span verification roll-up to AIBEST/STATUS.md once the chain finishes (so the record lands even if the
# coding session has ended). One write.
L=/mnt/models/bl1013/AIBEST/logs
S=/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/BACKLOG_RUN_20261013/AIBEST/STATUS.md
until grep -q "chain done" $L/s_summary.log 2>/dev/null; do sleep 60; done
{
  printf '\n[code] %s CDT AIBEST S format (92246e8de) verification chain finished. Roll-up:\n' "$(date '+%H:%M')"
  sed 's/^/    /' $L/s_summary.log
  grep -hE "T6|^PASS|^FAIL" $L/s_ratetest.log 2>/dev/null | grep -E "T6|failures" | sed 's/^/    /'
  grep -hE "ucheck:|FAIL" $L/s_ucheck.log 2>/dev/null | tail -4 | sed 's/^/    /'
  grep -hE "tests passed|FAIL" $L/s_bops_mm.log $L/s_bops_mmid.log 2>/dev/null | tail -6 | sed 's/^/    /'
  grep -h "selftest" $L/s_self_*.log 2>/dev/null | sed 's/^/    /'
  grep -hE "wrote |not constant" $L/s_repack.log 2>/dev/null | sed 's/^/    /'
} >> $S
