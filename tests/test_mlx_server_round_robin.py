import os
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed .*socket")

from tests.mlx_server_test_utils import (
    extract_token_count,
    post_json,
    start_mlx_servers,
    start_nginx_round_robin,
)


class MLXServerRoundRobinTest(unittest.TestCase):
    def test_round_robin_throughput(self):
        prompt = os.environ.get(
            "MLX_PROMPT",
            "Share three optimization tips for model serving.",
        )
        max_tokens = int(os.environ.get("MLX_MAX_TOKENS", "96"))
        temperature = float(os.environ.get("MLX_TEMPERATURE", "0.3"))
        instance_count = int(os.environ.get("MLX_SERVER_INSTANCES", "4"))
        base_port = int(os.environ.get("MLX_SERVER_BASE_PORT", "9000"))
        nginx_port = int(os.environ.get("MLX_NGINX_PORT", "8088"))

        total_requests_env = os.environ.get("MLX_NUM_REQUESTS")
        if total_requests_env:
            total_requests = int(total_requests_env)
        else:
            total_requests = max(16, instance_count * 16)

        concurrency_env = os.environ.get("MLX_CONCURRENCY", "max")
        if concurrency_env.lower() in {"max", "maximum", "all"}:
            concurrency = total_requests
        else:
            concurrency = int(concurrency_env)

        if instance_count < 1:
            instance_count = 1
        if total_requests < 1:
            total_requests = 1
        if concurrency < 1:
            concurrency = 1
        if concurrency > total_requests:
            concurrency = total_requests

        with start_mlx_servers(
            instance_count,
            base_port=base_port,
        ) as servers:
            upstreams = [(server["host"], server["port"]) for server in servers]
            with start_nginx_round_robin(
                upstreams,
                listen_port=nginx_port,
                listen_host=servers[0]["host"],
            ) as proxy:
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            post_json,
                            f"{proxy['base_url']}/v1/chat/completions",
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
            "round_robin "
            f"instances={instance_count} "
            f"requests={total_requests} concurrency={concurrency} "
            f"total_tokens={total_tokens} "
            f"elapsed={total_time:.2f}s "
            f"throughput_tps={throughput:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
