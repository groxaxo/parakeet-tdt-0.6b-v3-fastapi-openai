#!/usr/bin/env bash
# Pin the Parakeet v3 server to the Intel 12th Gen P-cores (cores 0-7, threads 0-15).
# Usage:  ./pin_pcores.sh python server.py
#         ./pin_pcores.sh uvicorn parakeet_service.main:app --port 5092
set -euo pipefail

# Detect P-cores via /sys (Intel hybrid). Fallback to physical-core assumption.
PCORES=""
if [[ -d /sys/devices/cpu_core ]]; then
    PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || true)
fi
if [[ -z "$PCORES" ]]; then
    # Conservative default for 12700K/12700KF: P-cores = logical 0-15
    PCORES="0-15"
fi

echo "Pinning to P-cores: $PCORES" >&2
exec taskset -c "$PCORES" "$@"
