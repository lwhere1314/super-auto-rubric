import unittest
from unittest.mock import patch

import requests

from super_auto_rubric.webshop.seed_plan import SeedPlanChatClient


def make_response(status_code, body):
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


class SeedPlanChatClientTest(unittest.TestCase):
    def test_retries_retryable_status_codes(self):
        throttled = make_response(429, '{"error":{"message":"rate limited"}}')
        ok = make_response(200, '{"choices":[{"message":{"content":"done"}}]}')
        client = SeedPlanChatClient(
            api_key="test",
            base_url="https://example.com",
            max_retries=1,
            retry_base_sleep_seconds=0,
        )

        with patch("requests.post", side_effect=[throttled, ok]) as mock_post:
            completion = client.complete(model="m", messages=[])

        self.assertEqual(completion.content, "done")
        self.assertEqual(mock_post.call_count, 2)

    def test_does_not_retry_non_retryable_status_codes(self):
        bad_request = make_response(400, '{"error":{"message":"bad request"}}')
        client = SeedPlanChatClient(
            api_key="test",
            base_url="https://example.com",
            max_retries=3,
            retry_base_sleep_seconds=0,
        )

        with patch("requests.post", return_value=bad_request) as mock_post:
            with self.assertRaises(requests.HTTPError):
                client.complete(model="m", messages=[])

        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
