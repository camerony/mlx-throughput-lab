import os
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning)

from tests.mlx_server_test_utils import (
    extract_token_count,
    post_json,
    start_mlx_server,
)


class MLXServerConcurrentRequestTest(unittest.TestCase):
    def test_concurrent_requests_throughput(self):
        prompt = os.environ.get(
            "MLX_PROMPT",
            "List five ways to make inference servers faster.",
        )
        max_tokens = int(os.environ.get("MLX_MAX_TOKENS", "96"))
        temperature = float(os.environ.get("MLX_TEMPERATURE", "0.3"))
        total_requests_env = os.environ.get("MLX_NUM_REQUESTS")
        if total_requests_env:
            total_requests = int(total_requests_env)
        else:
            cpu_count = os.cpu_count() or 1
            total_requests = max(8, cpu_count * 4)

        concurrency_env = os.environ.get("MLX_CONCURRENCY", "max")
        if concurrency_env.lower() in {"max", "maximum", "all"}:
            concurrency = total_requests
        else:
            concurrency = int(concurrency_env)

        if total_requests < 1:
            total_requests = 1
        if concurrency < 1:
            concurrency = 1
        if concurrency > total_requests:
            concurrency = total_requests

        with start_mlx_server() as server:
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        post_json,
                        f"{server['base_url']}/v1/chat/completions",
                        {
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stream": False,
                        },
                    )
                    for _ in range(total_requests)
                ]
                results = [future.result() for future in as_completed(futures)]
            total_time = time.time() - start_time

        total_tokens = sum(extract_token_count(result) for result in results)
        throughput = total_tokens / total_time if total_time > 0 else 0.0

        self.assertGreater(total_tokens, 0, "Expected tokens from responses.")
        self.assertGreater(throughput, 0.0, "Expected throughput > 0.")
        print(
            "concurrent_requests "
            f"count={total_requests} concurrency={concurrency} "
            f"total_tokens={total_tokens} "
            f"elapsed={total_time:.2f}s "
            f"throughput_tps={throughput:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
