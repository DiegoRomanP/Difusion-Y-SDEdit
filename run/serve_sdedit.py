# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "diffusers==0.42.1", "safetensors", "numpy", "pillow"]
# ///
"""Servidor local de SDEdit para la presentación web (usa la GPU si hay, si no la CPU).

    uv run serve_sdedit.py ruta/a/best.safetensors

Escucha solo en 127.0.0.1:8765. La presentación le envía una imagen y recibe el proceso
completo (original → ruido → imagen nueva) como PNG en base64 más un GIF. Responde:
  GET  /health   → {"ok": true, "device": ...}
  POST /sdedit   → cuerpo = bytes de la imagen; ?strength=0.6&seed=12345 opcionales
"""

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pc3.data import canonical_image  # noqa: E402
from pc3.inference import sdedit, to_tensor  # noqa: E402
from pc3.model import load_weights  # noqa: E402

MAX_BODY = 20 * 1024 * 1024
LOCK = threading.Lock()  # un SDEdit a la vez: la red es grande y el equipo es uno solo


def data_url(image: Image.Image, fmt: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def make_handler(model, device, default_strength, steps):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._send(200, {"ok": True, "device": str(device)})
            else:
                self._send(404, {"error": "no encontrado"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/sdedit":
                return self._send(404, {"error": "no encontrado"})
            query = parse_qs(url.query)
            try:
                strength = float(query.get("strength", [default_strength])[0])
                seed = int(query.get("seed", [12345])[0])
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < strength <= 1 or not 0 < length <= MAX_BODY:
                    raise ValueError
                photo = canonical_image(io.BytesIO(self.rfile.read(length)))
            except Exception:
                return self._send(400, {"error": "imagen o parámetros no válidos"})
            with LOCK:
                _, frames = sdedit(
                    model, to_tensor(photo).unsqueeze(0), strength=strength, steps=steps,
                    seed=seed, device=device, capture=True,
                )
            sequence = [photo, *frames]
            gif = io.BytesIO()
            sequence[0].save(gif, "GIF", save_all=True, append_images=sequence[1:], duration=200, loop=0)
            self._send(200, {
                "frames": [data_url(f, "PNG") for f in sequence],
                "gif": "data:image/gif;base64," + base64.b64encode(gif.getvalue()).decode(),
            })

        def log_message(self, fmt, *args):
            print(self.address_string(), fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("weights", type=Path, help="archivo best.safetensors")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--strength", type=float, default=0.6, help="fuerza por defecto (0-1]")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"no existe el archivo de pesos: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_weights(args.weights)
    model.to(device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, device, args.strength, args.steps))
    print(f"SDEdit listo en http://127.0.0.1:{args.port} ({device}). Ctrl+C para salir.")
    server.serve_forever()


if __name__ == "__main__":
    main()
