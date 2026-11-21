"""
Test Segmentation Service

Smoke test against a real server process. It:
1. Writes a self-signed development certificate (the server only speaks TLS)
2. Starts `python -m segmentation_server` with that certificate
3. Waits until GetServerStatus answers over TLS
4. Runs the client example on an image (default: the RC1 crop in images/), trusting the certificate
5. Shuts the server down

Usage: python test_service.py [IMAGE] [--point X Y]

Needs a GPU machine with torch and SAM2 installed, like the server itself.
"""

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import grpc

from client_example import _channel_for, segment_image, show_segments
from segmentation_grpc import ServerStatusRequest, SegmentationServiceStub
from segmentation_server.dev_cert import generate_self_signed

STARTUP_TIMEOUT_SECONDS = 300


def _free_port() -> int:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
        probe.bind(("::", 0))
        return probe.getsockname()[1]


async def _wait_until_serving(address: str, ca_cert: str, process: subprocess.Popen) -> bool:
    """Poll GetServerStatus until it answers, the process exits, or the timeout passes."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            async with _channel_for(address, ca_cert=ca_cert) as channel:
                await SegmentationServiceStub(channel).GetServerStatus(
                    ServerStatusRequest(), timeout=5
                )
            return True
        except grpc.RpcError:
            await asyncio.sleep(2)
    return False


# A 1024 RC1 crop from training set v3 (section 111, location 1528873); the default point is inside that cell.
DEFAULT_IMAGE = Path(__file__).resolve().parent / "images" / "RC1_111_D1_X72192-73216_Y63488-64512.png"
DEFAULT_POINT = (603, 564)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE,
                        help=f"Image to segment (default: {DEFAULT_IMAGE.name}).")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), default=None,
                        help="Foreground click in pixels, x right and y down "
                             f"(default: {DEFAULT_POINT} for the default image, else the image centre).")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"no image at {args.image}")
    if args.point is None:
        if args.image.resolve() == DEFAULT_IMAGE:
            args.point = DEFAULT_POINT
        else:
            from PIL import Image
            with Image.open(args.image) as image:
                args.point = (image.width // 2, image.height // 2)
    args.point = tuple(args.point)
    return args


async def test_service(sample_image: Path, point: tuple):
    """Start the service over TLS, segment one image, and shut it down."""
    port = _free_port()
    with tempfile.TemporaryDirectory() as folder:
        cert, key = generate_self_signed(Path(folder) / "cert.pem", Path(folder) / "key.pem")
        environment = dict(os.environ, SSL_CERT_PATH=str(cert), SSL_KEY_PATH=str(key))

        print("Starting the service...")
        # A file, not a pipe: nothing reads a pipe while the test runs, and a server that logs
        # more than the pipe buffer holds would block on its next write.
        log_path = Path(folder) / "service.log"
        log_file = open(log_path, "w", encoding="utf-8")
        service_process = subprocess.Popen(
            [sys.executable, '-m', 'segmentation_server',
             '--tls-port', str(port), '--workers', '4', '--no-compile-image-encoder'],
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            address = f'localhost:{port}'
            print("Waiting for the service to answer over TLS...")
            if not await _wait_until_serving(address, str(cert), service_process):
                print("The service did not start.")
                return

            print(f"Running the client example with image {sample_image}, point {point}")
            image, segments = await segment_image(
                address,
                str(sample_image),
                [point],
                [1],
                ca_cert=str(cert),
            )

            if image is not None and segments is not None:
                print("Segmentation successful!")
                print(f"Found {len(segments)} segments.")
                show_segments(image, segments, [point], [1])
            else:
                print("Segmentation failed.")
        finally:
            print("Shutting down the service...")
            service_process.terminate()
            try:
                service_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                service_process.kill()
                service_process.wait()
            log_file.close()
            output = log_path.read_text(encoding="utf-8", errors="replace")
            if output:
                print("Service output:")
                print(output)


if __name__ == '__main__':
    arguments = _parse_args()
    asyncio.run(test_service(arguments.image, arguments.point))
