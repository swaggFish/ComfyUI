"""
ComfyUI WebSocket/REST API Client

Low-level client for communicating with a running ComfyUI instance.
Based on the official websockets_api_example.py, extended with:
- Retry logic and connection management
- Progress callbacks
- Video/image output download
- Image upload for FLF2V conditioning
"""

import json
import uuid
import time
import urllib.request
import urllib.parse
import urllib.error
import os
import logging
from pathlib import Path
from typing import Callable, Any

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Raised when ComfyUI returns an error or is unreachable."""
    pass


class ComfyUIClient:
    """Client for the ComfyUI WebSocket/REST API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188,
                 timeout: int = 600, max_retries: int = 3):
        """
        Args:
            host: ComfyUI server hostname
            port: ComfyUI server port
            timeout: Maximum seconds to wait for a single generation
            max_retries: Number of connection retry attempts
        """
        self.host = host
        self.port = port
        self.server_address = f"{host}:{port}"
        self.client_id = str(uuid.uuid4())
        self.timeout = timeout
        self.max_retries = max_retries
        self._ws = None

    # ── Connection Management ──────────────────────────────────────────

    def _ensure_websocket(self):
        """Ensure WebSocket connection is alive, reconnect if needed."""
        if websocket is None:
            raise ComfyUIError(
                "websocket-client is not installed. "
                "Install it with: pip install websocket-client"
            )

        if self._ws is not None:
            try:
                self._ws.ping()
                return
            except Exception:
                self._close_websocket()

        for attempt in range(self.max_retries):
            try:
                self._ws = websocket.WebSocket()
                self._ws.settimeout(self.timeout)
                self._ws.connect(
                    f"ws://{self.server_address}/ws?clientId={self.client_id}"
                )
                logger.info(f"Connected to ComfyUI at {self.server_address}")
                return
            except Exception as e:
                logger.warning(
                    f"WebSocket connect attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                time.sleep(2 ** attempt)

        raise ComfyUIError(
            f"Could not connect to ComfyUI at {self.server_address} "
            f"after {self.max_retries} attempts. Is ComfyUI running?"
        )

    def _close_websocket(self):
        """Safely close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def is_alive(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            url = f"http://{self.server_address}/system_stats"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def close(self):
        """Close all connections."""
        self._close_websocket()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── REST API ───────────────────────────────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """
        Submit a workflow (API format) to ComfyUI for execution.

        Args:
            workflow: The workflow dict in ComfyUI API format

        Returns:
            prompt_id: UUID string identifying this execution
        """
        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
            "prompt_id": prompt_id,
        }
        # Log payload for debugging
        with open("debug_prompt.json", "w") as f:
             json.dump(payload, f, indent=2)
        
        data = json.dumps(payload).encode('utf-8')
        url = f"http://{self.server_address}/prompt"

        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                if "error" in result:
                    raise ComfyUIError(f"Queue error: {result['error']}")
                logger.info(f"Queued prompt {prompt_id}")
                return prompt_id
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to queue prompt: {e}")

    def get_history(self, prompt_id: str) -> dict:
        """Fetch execution history/outputs for a completed prompt."""
        url = f"http://{self.server_address}/history/{prompt_id}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get(prompt_id, {})
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to get history: {e}")

    def download_output(self, filename: str, subfolder: str = "",
                        folder_type: str = "output") -> bytes:
        """Download a generated file (image/video) from ComfyUI."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        url_params = urllib.parse.urlencode(params)
        url = f"http://{self.server_address}/view?{url_params}"

        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to download {filename}: {e}")

    def upload_image(self, filepath: str, subfolder: str = "",
                     overwrite: bool = True) -> dict:
        """
        Upload an image to ComfyUI's input directory.

        Args:
            filepath: Local path to the image file
            subfolder: Subdirectory within ComfyUI's input folder
            overwrite: Whether to overwrite existing files

        Returns:
            dict with 'name', 'subfolder', 'type' keys
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")

        # Build multipart form data
        boundary = uuid.uuid4().hex
        filename = filepath.name

        body = bytearray()

        # Image file part
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{filename}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(filepath.read_bytes())
        body.extend(b"\r\n")

        # Overwrite field
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        )
        body.extend(str(overwrite).lower().encode())
        body.extend(b"\r\n")

        # Subfolder field (if specified)
        if subfolder:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                b'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            )
            body.extend(subfolder.encode())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode())

        url = f"http://{self.server_address}/upload/image"
        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                logger.info(f"Uploaded {filename} → {result}")
                return result
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to upload {filepath}: {e}")

    # ── WebSocket Execution Monitoring ─────────────────────────────────

    def execute_and_wait(self, workflow: dict,
                         on_progress: Callable[[dict], None] | None = None
                         ) -> dict:
        """
        Queue a workflow and wait for it to complete.

        Args:
            workflow: The workflow dict in ComfyUI API format
            on_progress: Optional callback receiving progress messages:
                         {"type": "progress", "node": str, "value": int, "max": int}
                         {"type": "executing", "node": str}
                         {"type": "status", "queue_remaining": int}

        Returns:
            dict mapping node_id → list of output info dicts
        """
        self._ensure_websocket()
        prompt_id = self.queue_prompt(workflow)

        logger.info(f"Waiting for execution of {prompt_id}...")

        start_time = time.time()
        while True:
            if time.time() - start_time > self.timeout:
                raise ComfyUIError(
                    f"Execution timed out after {self.timeout}s"
                )

            try:
                out = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                raise ComfyUIError("WebSocket receive timed out")
            except Exception as e:
                raise ComfyUIError(f"WebSocket error: {e}")

            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get("type", "")

                if msg_type == "executing":
                    data = message["data"]
                    if on_progress:
                        on_progress({
                            "type": "executing",
                            "node": data.get("node"),
                            "prompt_id": data.get("prompt_id"),
                        })
                    # Execution complete when node is None for our prompt
                    if (data.get("node") is None and
                            data.get("prompt_id") == prompt_id):
                        break

                elif msg_type == "progress":
                    data = message["data"]
                    if on_progress:
                        on_progress({
                            "type": "progress",
                            "node": data.get("node"),
                            "value": data.get("value", 0),
                            "max": data.get("max", 0),
                        })

                elif msg_type == "status":
                    data = message.get("data", {})
                    queue_remaining = data.get("status", {}).get(
                        "exec_info", {}
                    ).get("queue_remaining", -1)
                    if on_progress:
                        on_progress({
                            "type": "status",
                            "queue_remaining": queue_remaining,
                        })

                elif msg_type == "execution_error":
                    data = message.get("data", {})
                    raise ComfyUIError(
                        f"Execution error in node {data.get('node_id')}: "
                        f"{data.get('exception_message', 'Unknown error')}"
                    )
            else:
                # Binary data = latent preview frames, skip
                continue

        # Fetch results
        history = self.get_history(prompt_id)
        outputs = history.get("outputs", {})
        return outputs

    def download_all_outputs(self, outputs: dict,
                              output_dir: str) -> dict[str, list[str]]:
        """
        Download all output files from an execution result.

        Args:
            outputs: The outputs dict from execute_and_wait()
            output_dir: Local directory to save files to

        Returns:
            dict mapping node_id → list of local file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = {}
        for node_id, node_output in outputs.items():
            downloaded[node_id] = []

            # Handle images
            for image_info in node_output.get("images", []):
                data = self.download_output(
                    image_info["filename"],
                    image_info.get("subfolder", ""),
                    image_info.get("type", "output"),
                )
                local_path = output_dir / image_info["filename"]
                local_path.write_bytes(data)
                downloaded[node_id].append(str(local_path))
                logger.info(f"Downloaded image: {local_path}")

            # Handle videos/gifs
            for vid_info in node_output.get("gifs", []):
                data = self.download_output(
                    vid_info["filename"],
                    vid_info.get("subfolder", ""),
                    vid_info.get("type", "output"),
                )
                local_path = output_dir / vid_info["filename"]
                local_path.write_bytes(data)
                downloaded[node_id].append(str(local_path))
                logger.info(f"Downloaded video: {local_path}")

        return downloaded
