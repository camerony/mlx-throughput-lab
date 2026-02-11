import os
import sys
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed .*socket")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.mlx_server_test_utils import (
    extract_token_count,
    parse_comma_args,
    post_json,
    start_mlx_server,
)


def _parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


class MLXServerConcurrencySweepTest(unittest.TestCase):
    def test_concurrency_sweep_throughput(self):
        prompt = os.environ.get(
            "MLX_PROMPT",
            "List five ways to make inference servers faster.",
        )
        max_tokens = int(os.environ.get("MLX_MAX_TOKENS", "96"))
        temperature = float(os.environ.get("MLX_TEMPERATURE", "0.3"))
        total_requests = int(os.environ.get("MLX_NUM_REQUESTS", "8"))
        concurrency = int(os.environ.get("MLX_CONCURRENCY", "4"))

        decode_concurrency_list = _parse_int_list(
            os.environ.get("MLX_DECODE_CONCURRENCY_LIST"),
            "8,16,32,64",
        )
        prompt_concurrency_list = _parse_int_list(
            os.environ.get("MLX_PROMPT_CONCURRENCY_LIST"),
            "2,4,8,16",
        )
        base_args = parse_comma_args(os.environ.get("MLX_SERVER_ARGS", ""))

        best = {"throughput": 0.0, "decode_concurrency": None, "prompt_concurrency": None}

        for decode_conc in decode_concurrency_list:
            for prompt_conc in prompt_concurrency_list:
                extra_args = base_args + [
                    "--decode-concurrency", str(decode_conc),
                    "--prompt-concurrency", str(prompt_conc),
                ]

                with start_mlx_server(extra_args=extra_args) as server:
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
                        results = [
                            future.result() for future in as_completed(futures)
                        ]
                    total_time = time.time() - start_time

                total_tokens = sum(
                    extract_token_count(result) for result in results
                )
                throughput = total_tokens / total_time if total_time > 0 else 0.0

                self.assertGreater(
                    total_tokens, 0, "Expected tokens from responses."
                )
                self.assertGreater(
                    throughput, 0.0, "Expected throughput > 0."
                )

                if throughput > best["throughput"]:
                    best = {
                        "throughput": throughput,
                        "decode_concurrency": decode_conc,
                        "prompt_concurrency": prompt_conc,
                    }

                print(
                    "concurrency_sweep "
                    f"decode_concurrency={decode_conc} "
                    f"prompt_concurrency={prompt_conc} "
                    f"requests={total_requests} concurrency={concurrency} "
                    f"total_tokens={total_tokens} "
                    f"elapsed={total_time:.2f}s "
                    f"throughput_tps={throughput:.2f}"
                )

        print(
            "concurrency_sweep_best "
            f"decode_concurrency={best['decode_concurrency']} "
            f"prompt_concurrency={best['prompt_concurrency']} "
            f"throughput_tps={best['throughput']:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
