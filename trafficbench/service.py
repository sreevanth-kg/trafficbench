"""Small local HTTP demo for the exported forecast bundle."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from .inference import ForecastEngine


def make_handler(engine):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            self.respond(200, {"status": "ready", "architecture": engine.metadata["architecture"],
                               "sensors": len(engine.metadata["sensor_ids"]), "history_minutes": 60,
                               "forecast_minutes": engine.metadata["forecast_minutes"]})

        def do_POST(self):
            if self.path != "/forecast":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Request body must be between 1 byte and 1 MB.")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object.")
                with lock:
                    result = engine.predict(request)
                self.respond(200, result)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.respond(400, {"error": str(error)})

    return Handler


def run(args):
    engine = ForecastEngine(args.bundle, args.device, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"Forecast demo at http://{args.host}:{args.port}; GET /health, POST /forecast", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
