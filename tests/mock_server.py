"""A deterministic OpenAI-compatible streaming server; no external inference."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextmanager
def mock_model(*, hang_after_write=False, models_status=200, squid_error=False):
    requests = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            data = json.dumps({"object": "list", "data": [{"id": "mock/model"}]}).encode()
            self.send_response(models_status)
            if squid_error:
                self.send_header("X-Squid-Error", "ERR_ACCESS_DENIED 0")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"body": body, "authorization": self.headers.get("Authorization")})
            turn = len(requests)
            if hang_after_write and turn > 1:
                release.wait(90)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def event(delta=None, finish=None, usage=None):
                chunk = {
                    "id": f"mock-{turn}",
                    "object": "chat.completion.chunk",
                    "created": 1700000000 + turn,
                    "model": "mock/model",
                    "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
                }
                if usage is not None:
                    chunk["choices"] = []
                    chunk["usage"] = usage
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()

            if turn <= 2:
                args = {
                    "path": "answer.txt" if turn == 1 else "new.txt",
                    "content": "solved\n" if turn == 1 else "new\n",
                }
                name = "write"
            elif turn == 3:
                # The inference host is allowed, but other destinations AND other
                # ports on the allowed host must be denied by the proxy.
                port = self.server.server_port
                args = {
                    "command": (
                        "test ! -e /tests/test.sh && "
                        "test ! -e /solution && "
                        'test "$(curl -s -o /dev/null --max-time 5 --proxy "$http_proxy" '
                        f"--noproxy '' -w '%{{http_code}}' http://example.com:{port}/)\" = 403 && "
                        'test "$(curl -s -o /dev/null --max-time 5 --proxy "$http_proxy" '
                        f"--noproxy '' -w '%{{http_code}}' "
                        f'http://host.docker.internal:{port + 1}/)" '
                        "= 403 && echo isolation-ok"
                    )
                }
                name = "bash"
            else:
                event({"role": "assistant", "content": "Solved the fixture."})
                event(finish="stop")
                event(usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            event(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-{turn}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            event(finish="tool_calls")
            event(usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
