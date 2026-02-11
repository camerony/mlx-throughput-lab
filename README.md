# MLX Throughput Lab

Performance benchmarking framework for testing `mlx_lm.server` throughput on Apple Silicon.

## Quick Start

1) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Install dialog for interactive launcher:
```bash
brew install dialog  # macOS
```

4) Run the launcher:
```bash
./run_mlx_tests.py
```

The launcher lets you pick a test/sweep, select an MLX model, and enter env overrides.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) - MLX requires Metal GPU
- **Python 3.11+** - MLX requires recent Python
- **mlx-lm** - Install via `pip install mlx-lm`
- **nginx** - For round-robin tests/sweeps (`brew install nginx`)
- **dialog** - For interactive launcher (`brew install dialog`)
- **MLX Model** - HF repo name (e.g., `mlx-community/Mistral-7B-Instruct-v0.3-4bit`) or local path

## Run With Launcher

Use the interactive menu to pick tests or sweeps and supply optional env overrides.

The launcher accepts both:
- **Hugging Face model repos**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Local MLX model paths**: `./models/my-mlx-model` or paths in `~/.cache/huggingface/hub/`

Model auto-detection searches: `./models`, `~/models`, `~/Models`, `~/.cache/huggingface/hub/`

If you need to access the server from another machine, set `MLX_SERVER_HOST=0.0.0.0`
so nginx and mlx_lm.server bind to all interfaces (default is `127.0.0.1`).

```bash
./run_mlx_tests.py
```

## Run Directly (No Launcher)

Run any test or sweep directly with Python and environment variables.

```bash
# Single request test
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
python -m unittest tests/test_mlx_server_single.py

# Concurrent requests test
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
MLX_CONCURRENCY=32 MLX_NUM_REQUESTS=32 \
python -m unittest tests/test_mlx_server_concurrent.py

# Round-robin test (requires nginx)
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
MLX_SERVER_INSTANCES=2 MLX_CONCURRENCY=64 \
python -m unittest tests/test_mlx_server_round_robin.py

# Concurrency sweep
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
python tests/test_mlx_server_concurrency_sweep.py

# Full sweep
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
python scripts/full_sweep.py
```

## Launcher Options

Tests are quick, pass/fail checks you can run like normal unit tests. Sweeps are longer
benchmark runs that explore parameter ranges and report the best throughput.

**Tests:**
- Single request
- Concurrent requests
- Round-robin (nginx + multiple servers, requires `nginx`)

**Sweeps:**
- Concurrency (decode-concurrency × prompt-concurrency)
- Full (instances × decode-concurrency × prompt-concurrency × request-concurrency, requires `nginx`)

These use env overrides (e.g., `MLX_SERVER_INSTANCES`, `MLX_DECODE_CONCURRENCY`,
`MLX_PROMPT_CONCURRENCY`, `MLX_SERVER_BASE_PORT`, `MLX_NGINX_PORT`, `MLX_SERVER_HOST`,
`MLX_MODEL_PATH`).

## Environment Variables

You can supply overrides in the launcher (space-separated `KEY=VALUE` pairs), or set them in your shell.

### Core Paths

- `MLX_MODEL_PATH`: Model identifier - HF repo name (e.g., `mlx-community/Mistral-7B-Instruct-v0.3-4bit`) or local MLX model directory path.

### Server Behavior

- `MLX_SERVER_ARGS`: extra args passed to `mlx_lm.server` using **comma-separated** format with `=` for values (e.g. `--max-tokens=256,--temp=0.5,--chat-template=/path/to/template.txt`). Space-separated args are still accepted for backward compatibility but **won't handle paths with spaces**. Can be set via the **Advanced Args** menu in the launcher or via Env Overrides. If `--decode-concurrency` or `--prompt-concurrency` appear in these args, the corresponding computed value is **skipped entirely** (not duplicated).
- `MLX_DECODE_CONCURRENCY`: Parallel batch decode requests (default 32). Controls how many concurrent decode operations the server handles.
- `MLX_PROMPT_CONCURRENCY`: Parallel batch prompt processing (default 8). Controls concurrent prompt processing operations.
- `MLX_SERVER_HOST`: host for mlx_lm.server (default `127.0.0.1`).
- `MLX_SERVER_PORT`: fixed port for single-server tests (optional).
- `MLX_SERVER_INSTANCES`: number of servers for round-robin tests/sweeps.
- `MLX_SERVER_BASE_PORT`: base port for multi-server runs (default `9000`).
- `MLX_NGINX_PORT`: nginx listen port (default `8088`).
- `MLX_READY_TIMEOUT`: seconds to wait for model readiness.
- `MLX_SERVER_BIND_TIMEOUT`: seconds to wait for server to bind (default 180; increase if model load is slow).
- `MLX_STARTUP_DELAY_S`: delay between starting servers (stagger startup).

### Request Controls

- `MLX_PROMPT`: prompt text.
- `MLX_MAX_TOKENS`: tokens to generate per request (default 128).
- `MLX_TEMPERATURE`: sampling temperature (default 0.3).
- `MLX_CONCURRENCY`: concurrent HTTP requests (tests).
- `MLX_NUM_REQUESTS`: total requests per run (tests/sweeps).

### Sweep Controls

- `MLX_DECODE_CONCURRENCY_LIST`: comma/space list for decode-concurrency sweep (default `8,16,32,64`).
- `MLX_PROMPT_CONCURRENCY_LIST`: list for prompt-concurrency sweep (default `2,4,8,16`).
- `MLX_CONCURRENCY_LIST`: list of HTTP request concurrencies (default `1,2,4,8,16,32,64,128,256`).
- `MLX_INSTANCES_LIST`: list of instance counts for full sweep (default `1,2,4`).
- `MLX_REQUESTS_MULTIPLIER`: if `MLX_NUM_REQUESTS` is unset, total requests = concurrency * multiplier.
- `MLX_CONTINUE_ON_ERROR`: set to `0` to stop on the first failing config (default continues).
- `MLX_REQUEST_TIMEOUT`: per-request timeout (seconds, default 120).
- `MLX_RETRY_ATTEMPTS`: retries for transient HTTP errors (default 8).
- `MLX_RETRY_SLEEP_S`: base retry backoff (seconds, default 0.5).
- `MLX_CELL_PAUSE_S`: pause between sweep cells (seconds).
- `MLX_WARMUP_REQUESTS`: warmup requests before a sweep run (default 2).
- `MLX_RESULTS_DIR`: base directory for sweep output files (default `results`).

## Advanced Server Arguments

The launcher exposes an **Advanced Args** field that lets you pass arbitrary flags directly to `mlx_lm.server`
via the `MLX_SERVER_ARGS` environment variable.

### Format

Use **comma-separated** tokens (recommended). Join flags and values with `=`:

```
--max-tokens=256,--temp=0.5,--chat-template=/path with spaces/template.txt
```

Each comma-delimited token becomes one argument. Paths with spaces are fully
supported because commas (not spaces) delimit arguments.

If you omit commas, the legacy space-separated format is still accepted, but
paths with spaces will break.

### Override behavior

If `--decode-concurrency` or `--prompt-concurrency` are present in `MLX_SERVER_ARGS`, the launcher
**skips injecting** the corresponding computed value entirely.

### Precedence: Env Overrides vs Advanced Args

If you set `MLX_SERVER_ARGS` in the **Env Overrides** field (e.g.
`MLX_SERVER_ARGS="--max-tokens=256"`), it takes priority over the Advanced Args
field. The Advanced Args value is only used when `MLX_SERVER_ARGS` is not
already present in Env Overrides.

### Sweep interaction

The full sweep (`scripts/full_sweep.py`) filters `--decode-concurrency` and `--prompt-concurrency`
from `MLX_SERVER_ARGS` and replaces them with sweep-specific values. Other arguments
(e.g. `--temp`, `--max-tokens`) are preserved.

### Limitations

- **Reserved flags in sweeps**: `--decode-concurrency` and `--prompt-concurrency` are
  auto-managed by sweep scripts and will be stripped/replaced. Do not rely on
  setting these through Advanced Args when running sweeps.

## Examples

### Launcher

Run the launcher and pass overrides in the dialog:
```
MLX_CONCURRENCY=64 MLX_NUM_REQUESTS=64 MLX_SERVER_ARGS="--decode-concurrency=64,--prompt-concurrency=16"
```

Note: `MLX_SERVER_ARGS` is for fixed runs. For the full sweep, use
`MLX_DECODE_CONCURRENCY_LIST` and `MLX_PROMPT_CONCURRENCY_LIST` (it already sweeps these),
so you can omit `MLX_SERVER_ARGS`.

### Direct

Run a concurrent test with custom concurrency and requests:
```bash
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
MLX_CONCURRENCY=64 MLX_NUM_REQUESTS=64 \
python -m unittest tests/test_mlx_server_concurrent.py
```

Run full sweep with custom ranges:
```bash
MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
MLX_INSTANCES_LIST="1,2,4" \
MLX_DECODE_CONCURRENCY_LIST="16,32,64" \
MLX_PROMPT_CONCURRENCY_LIST="4,8,16" \
MLX_CONCURRENCY_LIST="32,64,128" \
python scripts/full_sweep.py
```

Sweep results are written incrementally to:
```
results/full_sweep/full_sweep_<timestamp>.csv
```
Progress updates are printed to stderr during sweeps (completed/total and elapsed time).

## Analyze the Data

```bash
python analyze-data.py --file results/full_sweep/full_sweep_20260211_150913.csv --field throughput_tps --order desc --count 10
```

Parameters:
```plaintext
--file    ... which file you want to process (required)
--field   ... which field do you want to sort by (throughput_tps is the default if none is given)
--order   ... 'asc' or 'desc' for ascending or descending (descending is the default if not given)
--count   ... how many records to show (5 is the default)
```

Output will look something like this:

```plaintext
$ python analyze-data.py --file results/full_sweep/full_sweep_20260211_150913.csv
instances | decode_concurrency | prompt_concurrency | concurrency | throughput_tps | total_tokens | elapsed_s | errors
-----------------------------------------------------------------------------------------------------------------------------
      2.0 |               64.0 |               16.0 |       128.0 |          425.3 |      16384.0 |     38.52 |    0.0
      2.0 |               32.0 |                8.0 |        64.0 |          289.1 |       8192.0 |     28.33 |    0.0
      1.0 |               64.0 |               16.0 |        64.0 |          245.7 |       8192.0 |     33.34 |    0.0
      2.0 |               64.0 |                8.0 |       128.0 |          198.4 |       8448.0 |     42.58 |    4.0
      1.0 |               32.0 |                8.0 |        32.0 |          156.2 |       4096.0 |     26.21 |    0.0
```

## Popular MLX Models

Here are some recommended models from the mlx-community on Hugging Face:

**Small Models (faster, less memory):**
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Qwen2.5-3B-Instruct-4bit`

**Medium Models (good balance):**
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Llama-3.1-8B-Instruct-4bit`

**Large Models (best quality, more memory):**
- `mlx-community/Qwen2.5-14B-Instruct-4bit`
- `mlx-community/Llama-3.1-70B-Instruct-4bit` (requires significant GPU memory)

The `-4bit` suffix indicates 4-bit quantization, which reduces memory usage and often allows higher concurrency.

## Remote Testing

If you have a Mac Studio or other Apple Silicon Mac accessible via SSH (e.g., `criver@bigmac`), you can:

1. SSH to the remote machine and run tests directly:
```bash
ssh criver@bigmac
cd mlx-throughput-lab
source .venv/bin/activate
./run_mlx_tests.py
```

2. Or configure the server to listen on all interfaces for remote access:
```bash
export MLX_SERVER_HOST=0.0.0.0
```

## Differences from llama-throughput-lab

This framework is adapted from llama-throughput-lab with the following key changes:

1. **Server Invocation**: Uses `python -m mlx_lm.server` instead of `llama-server` binary
2. **API Format**: OpenAI-compatible `/v1/chat/completions` instead of `/completion`
3. **Request Format**: Messages array instead of prompt string
4. **Concurrency Parameters**: `--decode-concurrency` / `--prompt-concurrency` instead of `--parallel` / `--ctx-size`
5. **Model Format**: MLX models (HF repos or local) instead of GGUF files
6. **Platform**: Apple Silicon only (MLX requires Metal GPU)

## License

MIT License - Copyright (c) 2025

Adapted from llama-throughput-lab by Alex Ziskind.
