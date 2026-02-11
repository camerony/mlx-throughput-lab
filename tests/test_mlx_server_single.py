import os
import time
import unittest

from tests.mlx_server_test_utils import (
    extract_token_count,
    post_json,
    start_mlx_server,
)


class MLXServerSingleRequestTest(unittest.TestCase):
    def test_single_request_tokens_per_second(self):
        prompt = os.environ.get(
            "MLX_PROMPT",
            "Write a short paragraph about why concurrency helps throughput.",
        )
        max_tokens = int(os.environ.get("MLX_MAX_TOKENS", "128"))
        temperature = float(os.environ.get("MLX_TEMPERATURE", "0.2"))

        with start_mlx_server() as server:
            start_time = time.time()
            response = post_json(
                f"{server['base_url']}/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
            )
            elapsed = time.time() - start_time

        token_count = extract_token_count(response)
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0

        self.assertGreater(token_count, 0, "Expected tokens in response.")
        self.assertGreater(tokens_per_second, 0.0, "Expected tokens per second > 0.")
        print(
            "single_request "
            f"tokens={token_count} tokens_per_second={tokens_per_second:.2f} "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    unittest.main()
