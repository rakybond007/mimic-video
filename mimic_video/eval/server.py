"""
ZeroMQ-based policy server/client for distributed LIBERO evaluation.

Reference: gr00t/eval/service.py (BaseInferenceServer, BaseInferenceClient)
"""

import io
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

try:
    import msgpack
    import zmq

    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class MsgSerializer:
    """Serialize/deserialize dicts with numpy array support via MessagePack."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray__": True, "data": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if "__ndarray__" in obj:
            return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    ZeroMQ REP server serving MimicVideoPolicy.get_action().

    Usage:
        policy = MimicVideoPolicy.from_checkpoint(...)
        server = PolicyServer(policy, port=5555)
        server.run()
    """

    def __init__(self, policy, host: str = "*", port: int = 5555, api_token: str = None):
        if not HAS_ZMQ:
            raise ImportError("zmq and msgpack required. Install: pip install pyzmq msgpack")

        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.api_token = api_token

        self._endpoints: dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        self.register_endpoint(
            "get_modality_config", self._handle_get_modality_config, requires_input=False
        )

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def _kill_server(self):
        self.running = False
        return {"status": "ok", "message": "Server shutting down"}

    def _handle_get_action(self, data: dict) -> dict:
        return self.policy.get_action(data)

    def _handle_get_modality_config(self) -> dict:
        return self.policy.get_modality_config()

    def _validate_token(self, request: dict) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Policy server listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))

            except Exception as e:
                print(f"Server error: {e}")
                import traceback
                traceback.print_exc()
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    def close(self):
        self.socket.close()
        self.context.term()


class PolicyClient:
    """
    ZeroMQ REQ client connecting to PolicyServer.

    Usage:
        client = PolicyClient(host="localhost", port=5555)
        action = client.get_action(observation_dict)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        if not HAS_ZMQ:
            raise ImportError("zmq and msgpack required. Install: pip install pyzmq msgpack")

        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the server given observations."""
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> dict:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def kill_server(self):
        self.call_endpoint("kill", requires_input=False)

    def __del__(self):
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass
