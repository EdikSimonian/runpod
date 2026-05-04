#!/usr/bin/env bash
# Watchdog: probe SGLang /v1/models every 60s. After N consecutive failures,
# print a loud alert and ring the terminal bell. Optionally trigger fallback.
#
# Run from your laptop in the background:
#   bash workshop/scripts/watchdog.sh &
#
# To enable auto-fallback to H100, run with:
#   AUTO_FALLBACK=1 bash workshop/scripts/watchdog.sh &
#   (will invoke fallback-h100.sh after 3 consecutive failures)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/../lib/env.sh"
[ -f "$SCRIPT_DIR/../lib/env.local.sh" ] && . "$SCRIPT_DIR/../lib/env.local.sh"

require_var POD_TCP_HOST
require_var POD_TCP_PORT
require_var SGLANG_API_KEY

INTERVAL="${WATCHDOG_INTERVAL:-60}"   # seconds between probes
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-3}"
AUTO_FALLBACK="${AUTO_FALLBACK:-0}"

log "Watchdog starting: probing http://${POD_TCP_HOST}:${POD_TCP_PORT}/v1/models every ${INTERVAL}s"
log "  fail threshold: $FAIL_THRESHOLD  auto-fallback: $AUTO_FALLBACK"

consecutive_fail=0
while true; do
    if curl -sf -m 10 -H "Authorization: Bearer $SGLANG_API_KEY" \
        "http://${POD_TCP_HOST}:${POD_TCP_PORT}/v1/models" >/dev/null 2>&1; then
        if [ $consecutive_fail -gt 0 ]; then
            log "RECOVERED after $consecutive_fail consecutive failures"
            consecutive_fail=0
        fi
    else
        consecutive_fail=$((consecutive_fail + 1))
        log "PROBE FAILED ($consecutive_fail / $FAIL_THRESHOLD)"
        if [ $consecutive_fail -ge $FAIL_THRESHOLD ]; then
            printf '\a\a\a' >&2  # terminal bell × 3
            echo "" >&2
            echo "=== ALERT: SGLang unresponsive after $FAIL_THRESHOLD probes ===" >&2
            echo "  endpoint: http://${POD_TCP_HOST}:${POD_TCP_PORT}" >&2
            echo "  time: $(date)" >&2

            # macOS native notification (silent if not on macOS)
            command -v osascript >/dev/null 2>&1 \
                && osascript -e 'display notification "SGLang unresponsive — see watchdog log" with title "Workshop alert"' \
                || true

            if [ "$AUTO_FALLBACK" = "1" ]; then
                echo "  AUTO_FALLBACK=1 — invoking fallback-h100.sh..." >&2
                if bash "$SCRIPT_DIR/fallback-h100.sh"; then
                    echo "  Fallback succeeded — re-execing watchdog against new endpoint" >&2
                    # Re-exec with AUTO_FALLBACK=0 so we don't recurse if H100 also dies
                    exec env AUTO_FALLBACK=0 bash "$0"
                else
                    echo "  Fallback FAILED — exiting watchdog" >&2
                    exit 1
                fi
            fi

            # Reset counter so we don't spam alerts every probe
            consecutive_fail=0
        fi
    fi
    sleep "$INTERVAL"
done
