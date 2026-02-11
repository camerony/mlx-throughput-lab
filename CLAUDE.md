# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX Throughput Lab is a performance benchmarking framework for testing `mlx_lm.server` throughput under various configurations. It provides both an interactive dialog-based launcher (`./run_mlx_tests.py`) and direct Python scripts for running standardized throughput tests and parameter sweeps.

This is adapted from llama-throughput-lab but specifically designed for Apple Silicon machines running MLX (Metal Performance Shaders).

## Development Commands

### Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify mlx_lm is installed
python -c "import mlx_lm; print('MLX LM installed successfully')"
```

### Running Tests

```bash
# Interactive launcher (recommended for guided testing)
./run_mlx_tests.py

# Direct test execution
python -m unittest tests/test_mlx_server_single.py
python -m unittest tests/test_mlx_server_concurrent.py
python -m unittest tests/test_mlx_server_round_robin.py

# Run sweeps directly
python tests/test_mlx_server_concurrency_sweep.py
python scripts/full_sweep.py
```

### Linting

```bash
# Python syntax check
python -m compileall run_mlx_tests.py scripts tests

# Ruff linting (F and E9 rules only - fatal errors)
ruff check --select F,E9 .

# Shellcheck for bash scripts
shellcheck start_mlx_rr.sh
```

### Analyzing Results

```bash
# Analyze sweep CSV output
python analyze-data.py --file results/full_sweep/full_sweep_<timestamp>.csv --field throughput_tps --order desc --count 10
```

## Architecture

### Entry Point Structure

The codebase supports two execution modes:

1. **Interactive Mode**: `run_mlx_tests.py` provides a dialog-based menu system that guides users through test selection
2. **Direct Execution**: Tests and sweeps can be run directly via Python with environment variable configuration

### MLX Server Invocation

Unlike llama-throughput-lab which uses a binary (`llama-server`), this framework invokes `mlx_lm.server` as a Python module:

```python
subprocess.Popen([
    sys.executable, "-m", "mlx_lm.server",
    "--host", host,
    "--port", str(port),
    "--model", model_name,
    "--decode-concurrency", str(decode_concurrency),
    "--prompt-concurrency", str(prompt_concurrency),
])
```

### Model Detection

Located in [tests/mlx_server_test_utils.py](tests/mlx_server_test_utils.py):

- **Model Support**: Both Hugging Face repo names (e.g., `mlx-community/Mistral-7B-Instruct-v0.3-4bit`) and local MLX model paths
- **Local Detection**: Searches common paths (`./models`, `~/models`, `~/.cache/huggingface/hub`) for directories containing `config.json`
- **Check Function**: `check_mlx_lm_installed()` - verifies mlx_lm is importable before starting server

### Environment Variable Configuration

All tests/sweeps are controlled via environment variables. Key variables:

**Core Paths:**
- `MLX_MODEL_PATH` - Model identifier (HF repo name or local path to MLX model directory)

**Server Configuration:**
- `MLX_SERVER_ARGS` - Additional server arguments (comma-separated format, see parsing details below)
- `MLX_DECODE_CONCURRENCY` - Parallel batch decode requests (default 32)
- `MLX_PROMPT_CONCURRENCY` - Parallel batch prompt processing (default 8)
- `MLX_SERVER_INSTANCES` - Number of servers for round-robin tests
- `MLX_SERVER_BASE_PORT` - Base port for multi-server runs (default 9000)
- `MLX_NGINX_PORT` - Nginx listen port (default 8088)
- `MLX_SERVER_HOST` - Server host (default 127.0.0.1; use 0.0.0.0 for remote access)

**Request Controls:**
- `MLX_MAX_TOKENS` - Tokens to generate per request (default 128)
- `MLX_TEMPERATURE` - Sampling temperature (default 0.3)
- `MLX_CONCURRENCY` - Concurrent HTTP requests
- `MLX_NUM_REQUESTS` - Total requests per run

**Sweep Controls:**
- `MLX_INSTANCES_LIST` - Comma-separated instance counts (e.g., "1,2,4")
- `MLX_DECODE_CONCURRENCY_LIST` - decode-concurrency values to sweep (e.g., "8,16,32,64")
- `MLX_PROMPT_CONCURRENCY_LIST` - prompt-concurrency values to sweep (e.g., "2,4,8,16")
- `MLX_CONCURRENCY_LIST` - Request concurrency values to sweep

### MLX_SERVER_ARGS Parsing

**Critical Implementation Detail**: `MLX_SERVER_ARGS` uses **comma-separated format** (not space-separated) to support paths with spaces.

Implementation in [tests/mlx_server_test_utils.py:19-24](tests/mlx_server_test_utils.py#L19-L24):

```python
def parse_comma_args(raw_args):
    if not raw_args:
        return []
    if "," in raw_args:
        return [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return shlex.split(raw_args)  # Fallback for legacy space-separated
```

**Format**: `--max-tokens=256,--temp=0.5,--chat-template=/path with spaces/template.txt`

**Override Behavior**: When `--decode-concurrency` or `--prompt-concurrency` appear in `MLX_SERVER_ARGS`, the launcher **skips** injecting the corresponding computed value entirely.

### API Endpoint Differences

MLX server uses OpenAI-compatible endpoints:

**llama.cpp:**
- Endpoint: `/completion`
- Payload: `{"prompt": "...", "n_predict": 128}`

**MLX:**
- Endpoint: `/v1/chat/completions`
- Payload: `{"messages": [{"role": "user", "content": "..."}], "max_tokens": 128}`

### Concurrency Parameters

Unlike llama.cpp's `--parallel` and `--ctx-size`, MLX uses:

- `--decode-concurrency`: Number of parallel batch decode requests the server can handle
- `--prompt-concurrency`: Number of parallel batch prompt processing operations

These are **internal MLX parallelism** within one server process. Combined with multi-instance round-robin testing, you can explore both vertical scaling (higher concurrency per instance) and horizontal scaling (more instances).

### CSV Result Output

Sweep scripts write incremental results to timestamped CSV files:
- `results/full_sweep/full_sweep_<timestamp>.csv`
- CSV columns for full sweep: `instances,decode_concurrency,prompt_concurrency,concurrency,throughput_tps,total_tokens,elapsed_s,errors`

Progress updates are printed to stderr during sweeps (completed/total and elapsed time).

## Key Files

### [tests/mlx_server_test_utils.py](tests/mlx_server_test_utils.py)

Core shared utilities that all tests depend on:
- `parse_comma_args()` - Parses MLX_SERVER_ARGS with comma-separated format
- `check_mlx_lm_installed()` - Verifies mlx_lm package is available
- `resolve_model_path()` - Gets model path/name from environment
- `start_mlx_server()` - Context manager for server lifecycle (uses `python -m mlx_lm.server`)
- `post_json()` - HTTP request helper
- `extract_token_count()` - Extracts completion tokens from OpenAI-format response

### [tests/test_mlx_server_single.py](tests/test_mlx_server_single.py)

Single request throughput test - measures baseline performance with one request.

### [tests/test_mlx_server_concurrent.py](tests/test_mlx_server_concurrent.py)

Concurrent requests test - uses ThreadPoolExecutor to send multiple requests in parallel to one server instance.

### [tests/test_mlx_server_round_robin.py](tests/test_mlx_server_round_robin.py)

Round-robin test with nginx load balancing across multiple MLX server instances.

### [tests/test_mlx_server_concurrency_sweep.py](tests/test_mlx_server_concurrency_sweep.py)

Sweeps decode-concurrency × prompt-concurrency to find optimal internal parallelism settings.

### [scripts/full_sweep.py](scripts/full_sweep.py)

Comprehensive benchmark exploring: instances × decode-concurrency × prompt-concurrency × request-concurrency
- Filters `--decode-concurrency`, `--prompt-concurrency` from `MLX_SERVER_ARGS` and replaces with sweep values
- Preserves other arguments like `--temp`, `--max-tokens`
- Writes incremental CSV output

## Important Patterns

### Environment Variable Precedence

1. `MLX_SERVER_ARGS` in Env Overrides field takes priority over Advanced Args field
2. Advanced Args field is only used when `MLX_SERVER_ARGS` is not in Env Overrides
3. Computed values (decode-concurrency, prompt-concurrency) are skipped if explicitly set in `MLX_SERVER_ARGS`

### Reserved Flags in Sweeps

When running sweeps, do NOT set these flags in `MLX_SERVER_ARGS` - they are auto-managed:
- `--decode-concurrency` (sweep variable)
- `--prompt-concurrency` (sweep variable)

These will be stripped and replaced by sweep-specific values.

### Round-Robin Testing

Requires nginx installed. Tests use:
- Multiple mlx_lm.server instances on ports `MLX_SERVER_BASE_PORT` + offset
- Nginx upstream configuration with round-robin load balancing
- Each instance gets its own decode-concurrency and prompt-concurrency settings

**Use Cases:**
- Compare vertical scaling (1 instance, high concurrency) vs horizontal scaling (multiple instances, lower concurrency each)
- Test GPU memory pressure with multiple server processes
- Simulate distributed deployment scenarios

### Testing Requirements

- Python 3.11+ (MLX requires recent Python)
- `mlx-lm` package installed (`pip install mlx-lm`)
- Apple Silicon Mac (M1/M2/M3 etc.) for MLX Metal acceleration
- `nginx` installed for round-robin tests/sweeps (`brew install nginx` on macOS)
- `dialog` tool installed for interactive launcher (`brew install dialog` on macOS)
- MLX-compatible model (HF repo from mlx-community or locally converted)

### Remote Testing

The Mac Studio at `criver@bigmac` can be used for testing:

```bash
# Set server to bind to all interfaces for remote access
export MLX_SERVER_HOST=0.0.0.0

# Run tests from development machine, hitting bigmac
# (or run launcher/tests directly on bigmac via SSH)
```

### Token Calculation

MLX server returns OpenAI-compatible format with `usage` field:
```json
{
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 128,
    "total_tokens": 138
  }
}
```

Throughput calculation: `tokens_per_second = completion_tokens / elapsed_time`

(MLX server doesn't report tokens_per_second directly like llama.cpp, so we calculate it from elapsed time)

## Model Examples

**Hugging Face Models (recommended):**
```bash
export MLX_MODEL_PATH="mlx-community/Mistral-7B-Instruct-v0.3-4bit"
export MLX_MODEL_PATH="mlx-community/Llama-3.2-3B-Instruct-4bit"
export MLX_MODEL_PATH="mlx-community/Qwen2.5-7B-Instruct-4bit"
```

**Local Models:**
```bash
export MLX_MODEL_PATH="./models/mistral-7b-instruct-mlx"
export MLX_MODEL_PATH="~/.cache/huggingface/hub/models--mlx-community--Mistral-7B-Instruct-v0.3-4bit/snapshots/..."
```

## Performance Considerations

1. **GPU Memory**: MLX uses Metal (GPU) memory. Multiple instances may compete for GPU resources.
2. **Concurrency Tuning**: Start with decode-concurrency=32, prompt-concurrency=8 and adjust based on results.
3. **Batch Processing**: MLX handles batching internally via concurrency parameters.
4. **Quantization**: 4-bit quantized models (e.g., `-4bit` suffix) use less memory, allow higher concurrency.
