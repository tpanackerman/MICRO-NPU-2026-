#!/bin/bash
# Submit one Stage 2 accuracy / performance job to the Arena with an explicit server binary and extra environment variables, wait for it and save its log.
#   scripts/stage2/qkv_arena.sh accuracy|performance --binary PATH --name NAME --log-dir DIR [--test NAME] [--concurrency N] [--env "K=V K2=V2"] [--no-wait]
# Same payload as arena_test.sh; the only differences: the binary is given (no build), `env.sh` is uploaded and sourced by the entrypoint, the log is written to DIR/NAME.log.
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE"

MODE="${1:-}"
case "$MODE" in
    accuracy|performance) shift ;;
    *) echo "qkv_arena.sh: first argument must be accuracy or performance" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python3}"
HELPER="scripts/stage2/$MODE.py"
TEST=""
CONCURRENCY=1
BINARY=""
NAME=""
LOGDIR=""
ENVS=""
wait_for_result=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --test) TEST="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --log-dir) LOGDIR="$2"; shift 2 ;;
        --env) ENVS="$2"; shift 2 ;;
        --no-wait) wait_for_result=0; shift ;;
        *) echo "qkv_arena.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BINARY" ] && [ -f "$BINARY" ] || { echo "qkv_arena.sh: --binary PATH is required" >&2; exit 2; }
[ -n "$NAME" ] || { echo "qkv_arena.sh: --name is required" >&2; exit 2; }
[ -n "$LOGDIR" ] || { echo "qkv_arena.sh: --log-dir is required" >&2; exit 2; }
mkdir -p "$LOGDIR"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
"$PYTHON" "$HELPER" --prepare "$staging" ${TEST:+--test "$TEST"} --concurrency "$CONCURRENCY"

{
    echo "#!/bin/sh"
    for kv in $ENVS; do echo "export $kv"; done
} > "$staging/env.sh"
# entrypoint = the stock one with `env.sh` sourced first
{
    echo "#!/bin/sh"
    echo 'cd "$(dirname "$0")"'
    echo '[ -f ./env.sh ] && . ./env.sh && echo "qkv_arena env: $(grep export ./env.sh | tr "\n" " ")"'
    tail -n +2 scripts/stage2/entrypoint.sh
} > "$staging/remote_entrypoint.sh"
cp "$BINARY" "$staging/server"
chmod +x "$staging/remote_entrypoint.sh" "$staging/server"
echo "$MODE" > "$staging/mode"
tar -czf "$staging/stage2-payload.tar.gz" \
    -C "$CRATE/scripts/stage2" accuracy.py scoring.py suites.json performance.py performance_text.json \
    -C "$staging" mode selection.json -C "$CRATE" public-test
"$PYTHON" "$HELPER" --bundle-python "$staging/stage2-python.tar.gz"

log="$LOGDIR/$NAME.log"
{
    echo "# job name: $NAME"
    echo "# mode: $MODE test: ${TEST:-all} concurrency: $CONCURRENCY env: ${ENVS:-none}"
    echo "# binary: $BINARY md5 $(md5sum "$BINARY" | cut -d' ' -f1)"
    echo "# submitted: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log"

set +e
submit_output=$(furiosa-arena submit \
    "$staging/remote_entrypoint.sh" "$staging/server" "$staging/stage2-payload.tar.gz" "$staging/stage2-python.tar.gz" "$staging/env.sh" \
    --name "$NAME" --entrypoint remote_entrypoint.sh --timeout "${RNGD_SUBMIT_TIMEOUT:-70}" 2>&1)
submit_status=$?
set -e
echo "$submit_output" | tee -a "$log"
[ "$submit_status" -eq 0 ] || { echo "# submit failed (exit $submit_status)" >> "$log"; echo "qkv_arena.sh: submit failed for $NAME" >&2; exit 1; }
job=$(printf '%s\n' "$submit_output" | sed -n 's/.*submitted job \([0-9][0-9]*\).*/\1/p' | head -1)
[ -n "$job" ] || { echo "qkv_arena.sh: no job id" >&2; exit 1; }
echo "# job: $job" >> "$log"
[ "$wait_for_result" -eq 1 ] || { echo "submitted job $job"; exit 0; }

deadline=$(( SECONDS + ${RNGD_TIMEOUT:-5400} ))
state=""
while [ "$SECONDS" -lt "$deadline" ]; do
    status_output=$(furiosa-arena status "$job" 2>&1 || true)
    state=$(sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$status_output" | head -1 | tr '[:upper:]' '[:lower:]')
    case "$state" in succeeded|failed|completed|cancelled|canceled) break ;; esac
    sleep 20
done
echo "# final status: $state" >> "$log"
echo "$status_output" >> "$log"
furiosa-arena logs "$job" >> "$log" 2>&1 || true
echo "job $job $state -> $log"
