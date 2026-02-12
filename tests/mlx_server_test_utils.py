import contextlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def parse_comma_args(raw_args):
    if not raw_args:
        return []
    if "," in raw_args:
        return [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return shlex.split(raw_args)


def _has_flag(args, flag):
    """Check if *flag* (e.g. ``--decode-concurrency``) is present in *args*.

    Matches both ``--flag value`` and ``--flag=value`` forms.
    """
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _get_flag_value(args, flag):
    """Return value for *flag* (e.g. ``--max-tokens``) from args, or None."""
    for idx, arg in enumerate(args):
        if arg == flag:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def check_mlx_lm_installed():
    """Check if mlx_lm is importable/installed."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _find_mlx_models():
    """Search common locations for MLX model directories."""
    search_paths = [
        Path("./models"),
        Path("~/models").expanduser(),
        Path("~/Models").expanduser(),
        Path("~/.cache/huggingface/hub").expanduser(),
        Path("/models"),
    ]

    found_models = []
    for search_path in search_paths:
        if not search_path.exists():
            continue
        try:
            for item in search_path.iterdir():
                if item.is_dir():
                    # Check if it looks like an MLX model (has config.json or similar)
                    if (item / "config.json").exists():
                        found_models.append(str(item))
        except PermissionError:
            continue

    return found_models


def resolve_model_path():
    """Get model path/name from environment.

    Can be:
    - Hugging Face repo name (e.g., "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    - Local path to MLX model directory
    """
    return os.environ.get("MLX_MODEL_PATH", "")


def _pick_port(allow_env_port=True):
    env_port = os.environ.get("MLX_SERVER_PORT") if allow_env_port else None
    if env_port:
        return int(env_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return sock.getsockname()[1]


def _wait_for_server(host, port, timeout_s=None):
    if timeout_s is None:
        timeout_s = int(os.environ.get("MLX_SERVER_BIND_TIMEOUT", "180"))
    deadline = time.time() + timeout_s
    last_error = None
    health_url = f"http://{host}:{port}/health"
    models_url = f"http://{host}:{port}/v1/models"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in {200, 404}:
                return
            last_error = exc
        except Exception as exc:
            last_error = exc

        try:
            with urllib.request.urlopen(models_url, timeout=2) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in {200, 404}:
                return
            last_error = exc
        except Exception as exc:
            last_error = exc

        time.sleep(0.5)

    raise RuntimeError(
        f"Server did not become ready at {host}:{port} within {timeout_s}s: {last_error}. "
        "Check that the port is free (e.g. stop any round-robin servers) and that the model "
        "loads in time (try increasing MLX_SERVER_BIND_TIMEOUT or MLX_READY_TIMEOUT)."
    )


def _wait_for_completion_ready(host, port, timeout_s=120):
    deadline = time.time() + timeout_s
    last_error = None
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")

    while time.time() < deadline:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                time.sleep(0.5)
                continue
            with exc:
                data = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP error {exc.code}: {data}") from exc
        except Exception as exc:
            time.sleep(0.5)
            last_error = exc

    raise RuntimeError(f"Model did not become ready: {last_error}")


@contextlib.contextmanager
def start_mlx_server(port=None, host=None, extra_args=None, ready_timeout_s=None):
    if not check_mlx_lm_installed():
        raise ImportError(
            "mlx_lm is not installed. Install it with: pip install mlx-lm"
        )

    model_path = resolve_model_path()
    if not model_path:
        raise ValueError(
            "Model path not set. Set MLX_MODEL_PATH to a Hugging Face repo "
            "(e.g., 'mlx-community/Mistral-7B-Instruct-v0.3-4bit') or local path."
        )

    if host is None:
        host = os.environ.get("MLX_SERVER_HOST", DEFAULT_HOST)
    if port is None:
        port = _pick_port()
    else:
        port = int(port)
    if extra_args is None:
        extra_args = parse_comma_args(os.environ.get("MLX_SERVER_ARGS", ""))

    # Filter out unsupported concurrency flags from extra_args
    # Note: --decode-concurrency and --prompt-concurrency were removed in recent mlx-lm versions
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--decode-concurrency", "--prompt-concurrency"):
            # Skip this flag and its value
            skip_next = True
            continue
        if arg.startswith("--decode-concurrency=") or arg.startswith("--prompt-concurrency="):
            continue
        filtered_args.append(arg)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_path,
    ]
    cmd += filtered_args

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        bind_timeout = int(os.environ.get("MLX_SERVER_BIND_TIMEOUT", "180"))
        _wait_for_server(host, port, timeout_s=bind_timeout)
        completion_timeout = ready_timeout_s
        if completion_timeout is None:
            completion_timeout = int(os.environ.get("MLX_READY_TIMEOUT", "120"))
        _wait_for_completion_ready(host, port, timeout_s=completion_timeout)
        yield {
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "process": process,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextlib.contextmanager
def start_mlx_servers(
    count,
    base_port,
    host=None,
    extra_args=None,
    ready_timeout_s=None,
    startup_delay_s=None,
):
    if count < 1:
        raise ValueError("count must be >= 1")
    if base_port is None:
        base_port = _pick_port(allow_env_port=False)

    servers = []
    with contextlib.ExitStack() as stack:
        for index in range(count):
            port = base_port + index
            servers.append(
                stack.enter_context(
                    start_mlx_server(
                        port=port,
                        host=host,
                        extra_args=extra_args,
                        ready_timeout_s=ready_timeout_s,
                    )
                )
            )
            if startup_delay_s:
                time.sleep(startup_delay_s)
        yield servers


def resolve_nginx_bin():
    env_path = os.environ.get("NGINX_BIN")
    if env_path:
        return env_path
    return "nginx"


def _wait_for_port(host, port, timeout_s=20):
    deadline = time.time() + timeout_s
    last_error = None

    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((host, port))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)

    raise RuntimeError(f"Port {port} did not become ready: {last_error}")


@contextlib.contextmanager
def start_nginx_round_robin(upstreams, listen_port, listen_host=None):
    nginx_bin = resolve_nginx_bin()
    if not (os.path.isfile(nginx_bin) or shutil.which(nginx_bin)):
        raise FileNotFoundError(
            "nginx binary not found. Install nginx or set NGINX_BIN."
        )

    if listen_host is None:
        listen_host = DEFAULT_HOST

    temp_dir = tempfile.TemporaryDirectory()
    conf_path = os.path.join(temp_dir.name, "nginx.conf")
    upstream_lines = "\n".join(
        [f"        server {host}:{port};" for host, port in upstreams]
    )
    conf = (
        "worker_processes 1;\n"
        f"pid {temp_dir.name}/nginx.pid;\n"
        f"error_log {temp_dir.name}/error.log;\n"
        "events { worker_connections 1024; }\n"
        "http {\n"
        f"    access_log {temp_dir.name}/access.log;\n"
        "    upstream mlx_backend {\n"
        f"{upstream_lines}\n"
        "    }\n"
        "    server {\n"
        f"        listen {listen_host}:{listen_port};\n"
        "        location / {\n"
        "            proxy_pass http://mlx_backend;\n"
        "            proxy_http_version 1.1;\n"
        "            proxy_set_header Connection \"\";\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    with open(conf_path, "w", encoding="utf-8") as handle:
        handle.write(conf)

    process = subprocess.Popen(
        [nginx_bin, "-c", conf_path, "-p", temp_dir.name, "-g", "daemon off;"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_port(listen_host, listen_port)
        yield {
            "host": listen_host,
            "port": listen_port,
            "base_url": f"http://{listen_host}:{listen_port}",
            "process": process,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        temp_dir.cleanup()


def post_json(url, payload, timeout=120):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            resp.close()
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        with exc:
            data = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error {exc.code}: {data}") from exc


def extract_token_count(response):
    """Extract completion token count from mlx_lm response.

    MLX server returns OpenAI-compatible format with usage field.
    """
    usage = response.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])

    # Fallback: check top level
    if "completion_tokens" in response:
        return int(response["completion_tokens"])

    return 0


def extract_tokens_per_second(response):
    """Calculate tokens per second from mlx_lm response.

    MLX server doesn't directly report tokens_per_second,
    so we calculate it from usage and timing if available.
    """
    # Check if custom timing info is included
    timings = response.get("timings") or {}
    if "tokens_per_second" in timings:
        return float(timings["tokens_per_second"])

    # For now, return 0.0 - caller should calculate from elapsed time
    return 0.0
