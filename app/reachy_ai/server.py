from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_config
from .openai_client import ModelError
from .orchestrator import Orchestrator


class ApiHandler(BaseHTTPRequestHandler):
    orchestrator: Orchestrator

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(HTTPStatus.OK, self.orchestrator.health())
        elif self.path == "/health/deep":
            self._json(HTTPStatus.OK, self.orchestrator.health(deep=True))
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {
                "object": "list",
                "data": [{
                    "id": "reachy-auto",
                    "object": "model",
                    "owned_by": "reachy-ai",
                }],
            })
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/route":
                decision = self.orchestrator.route(payload.get("messages", []), payload.get("route"))
                self._json(HTTPStatus.OK, decision.to_dict())
            elif self.path == "/v1/chat/completions":
                response, _ = self.orchestrator.chat(payload)
                self._json(HTTPStatus.OK, response)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except ModelError as exc:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy AI smart routing proxy")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    orchestrator = Orchestrator(config)
    ApiHandler.orchestrator = orchestrator
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), ApiHandler)
    print(f"Reachy AI router: http://{config.listen_host}:{config.listen_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
