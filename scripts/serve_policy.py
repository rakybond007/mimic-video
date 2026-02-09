#!/usr/bin/env python3
"""
HTTP policy server for MimicVideo.

Loads a MimicVideoPolicy from a checkpoint and serves it via HTTP.
The client uses only Python stdlib (no extra packages needed).
Runs in the mimic_video conda environment (GPU required).

Usage:
    python scripts/serve_policy.py \
        --checkpoint_path ./checkpoints/test_nan_fix/checkpoint-100 \
        --port 5555

Reference: Isaac-GR00T-AlinVLA/scripts/serve_policy.py
"""

import argparse
import base64
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np


def _encode_value(obj):
    """Encode numpy arrays as base64 in JSON-serializable dicts."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": base64.b64encode(obj.tobytes()).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, dict):
        return {k: _encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    return obj


def _decode_value(obj):
    """Decode base64-encoded numpy arrays from JSON."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            data = base64.b64decode(obj["data"])
            return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {k: _decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_value(v) for v in obj]
    return obj


class PolicyHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes to policy.get_action()."""

    policy = None  # Set by main()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        request = _decode_value(json.loads(body))

        endpoint = request.get("endpoint", "get_action")

        try:
            if endpoint == "ping":
                result = {"status": "ok", "message": "Server is running"}
            elif endpoint == "get_action":
                obs = request.get("data", {})
                result = self.policy.get_action(obs)
                result = _encode_value(result)
            elif endpoint == "get_modality_config":
                result = self.policy.get_modality_config()
            else:
                result = {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}

        resp = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        # Suppress per-request logs
        pass


def main():
    parser = argparse.ArgumentParser(description="MimicVideo HTTP Policy Server")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--stats_path", type=str,
        default="/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta",
        help="Path to dataset with stats.json for normalization",
    )
    parser.add_argument("--port", type=int, default=5555, help="HTTP server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--denoising_steps", type=int, default=16, help="Sampling steps")
    args = parser.parse_args()

    # Load policy
    from mimic_video.policy import MimicVideoPolicy

    print(f"Loading policy from {args.checkpoint_path}...")
    print(f"Using stats from {args.stats_path}")
    t0 = time.time()
    policy = MimicVideoPolicy.from_checkpoint(
        args.checkpoint_path,
        stats_path=args.stats_path,
        device=args.device,
        denoising_steps=args.denoising_steps,
    )
    print(f"Policy loaded in {time.time() - t0:.1f}s")

    # Start HTTP server
    PolicyHandler.policy = policy
    server = HTTPServer((args.host, args.port), PolicyHandler)
    print(f"Server ready on http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
