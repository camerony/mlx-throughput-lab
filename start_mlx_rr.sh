#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MLX_MODEL_PATH:-}"
HOST="${MLX_SERVER_HOST:-127.0.0.1}"
INSTANCES="${MLX_SERVER_INSTANCES:-2}"
BASE_PORT="${MLX_SERVER_BASE_PORT:-9000}"
MLX_SERVER_ARGS="${MLX_SERVER_ARGS:-}"

NGINX_BIN="${NGINX_BIN:-nginx}"
NGINX_PORT="${MLX_NGINX_PORT:-8088}"

RUN_DIR="${RUN_DIR:-/tmp/mlx-rr}"
ENV_STATE_FILE="$RUN_DIR/mlx-rr.env"

# Find Python - prefer venv
find_python() {
  if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "$SCRIPT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}

PYTHON_BIN="$(find_python)"

start() {
  mkdir -p "$RUN_DIR"

  if [ -z "$MODEL_PATH" ]; then
    echo "Model path not set. Set MLX_MODEL_PATH." >&2
    exit 1
  fi
  if ! command -v "$NGINX_BIN" >/dev/null 2>&1; then
    echo "nginx not found: $NGINX_BIN" >&2
    exit 1
  fi

  # Parse args: comma-separated is recommended; space-separated is legacy.
  EXTRA_ARGS=()
  if [ -n "$MLX_SERVER_ARGS" ]; then
    if [[ "$MLX_SERVER_ARGS" == *","* ]]; then
      IFS=',' read -ra EXTRA_ARGS <<< "$MLX_SERVER_ARGS"
      # Trim whitespace from each element
      for i in "${!EXTRA_ARGS[@]}"; do
        EXTRA_ARGS[$i]="$(echo "${EXTRA_ARGS[$i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      done
    else
      set -f
      # shellcheck disable=SC2206
      EXTRA_ARGS=($MLX_SERVER_ARGS)
      set +f
    fi
  fi

  for ((i = 0; i < INSTANCES; i++)); do
    port=$((BASE_PORT + i))
    log="$RUN_DIR/mlx-${port}.log"
    "$PYTHON_BIN" -m mlx_lm server \
      --host "$HOST" \
      --port "$port" \
      --model "$MODEL_PATH" \
      "${EXTRA_ARGS[@]}" >"$log" 2>&1 &
    echo $! > "$RUN_DIR/mlx-${port}.pid"
  done

  upstream_lines=""
  for ((i = 0; i < INSTANCES; i++)); do
    port=$((BASE_PORT + i))
    upstream_lines="${upstream_lines}    server ${HOST}:${port};\n"
  done

  conf="$RUN_DIR/nginx.conf"
  {
    printf "worker_processes 1;\n"
    printf "pid %s/nginx.pid;\n" "$RUN_DIR"
    printf "error_log %s/nginx-error.log;\n" "$RUN_DIR"
    printf "events { worker_connections 1024; }\n"
    printf "http {\n"
    printf "  access_log %s/nginx-access.log;\n" "$RUN_DIR"
    printf "  upstream mlx_backend {\n%b  }\n" "$upstream_lines"
    printf "  server {\n"
    printf "    listen %s:%s;\n" "$HOST" "$NGINX_PORT"
    printf "    location / {\n"
    printf "      proxy_pass http://mlx_backend;\n"
    printf "      proxy_http_version 1.1;\n"
    printf "      proxy_set_header Connection \"\";\n"
    printf "    }\n  }\n}\n"
  } > "$conf"

  "$NGINX_BIN" -c "$conf" -p "$RUN_DIR" -g "daemon off;" >"$RUN_DIR/nginx.stdout" 2>&1 &
  echo $! > "$RUN_DIR/nginx.shell.pid"

  cat > "$ENV_STATE_FILE" <<EOF
HOST=$HOST
INSTANCES=$INSTANCES
BASE_PORT=$BASE_PORT
NGINX_PORT=$NGINX_PORT
EOF

  echo "Started ${INSTANCES} mlx_lm.server instances and nginx on http://${HOST}:${NGINX_PORT}"
}

stop() {
  if [ -f "$ENV_STATE_FILE" ]; then
    # shellcheck source=/dev/null
    . "$ENV_STATE_FILE"
  fi

  if [ -f "$RUN_DIR/nginx.pid" ]; then
    kill "$(cat "$RUN_DIR/nginx.pid")" 2>/dev/null || true
  fi
  if [ -f "$RUN_DIR/nginx.shell.pid" ]; then
    kill "$(cat "$RUN_DIR/nginx.shell.pid")" 2>/dev/null || true
  fi
  for pidfile in "$RUN_DIR"/mlx-*.pid; do
    [ -f "$pidfile" ] || continue
    kill "$(cat "$pidfile")" 2>/dev/null || true
  done

  if command -v lsof >/dev/null 2>&1; then
    inst="${INSTANCES:-0}"
    for ((i = 0; i < inst; i++)); do
      port=$((BASE_PORT + i))
      pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
      if [ -n "$pid" ]; then
        name=$(ps -p "$pid" -o comm= 2>/dev/null || true)
        case "$name" in
          *python*|*mlx*)
            kill "$pid" 2>/dev/null || true
            ;;
        esac
      fi
    done
  fi

  rm -f "$RUN_DIR"/nginx.pid "$RUN_DIR"/nginx.shell.pid "$RUN_DIR"/mlx-*.pid "$ENV_STATE_FILE" 2>/dev/null || true
  echo "Stopped."
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  *) echo "Usage: $0 [start|stop]" >&2; exit 1 ;;
esac
