"""Regenerate the seg_proto package from its copies of Viking's gRPC_Protos files.

Run from annotation-gallery:  python tools/gen_seg_proto.py
The copies of the .proto files are the source of truth here; refresh them from Viking's gRPC_Protos (commit recorded in
seg_proto/PROTO_COMMIT) before regenerating:
  Segmentation/SAM2/segmentation.proto -> seg_proto/segmentation.proto
  Common/geometry.proto                -> seg_proto/Common/geometry.proto
segmentation.proto imports "Common/geometry.proto", so seg_proto is the include root and Common a subpackage. protoc
writes absolute imports (``import segmentation_pb2``, ``from Common import geometry_pb2``); they are rewritten to
relative ones so the package imports as ``seg_proto`` wherever it is placed (the tests, the gallery image's /app).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from grpc_tools import protoc

root = Path(__file__).resolve().parents[1] / "seg_proto"
status = protoc.main(
    [
        "protoc", f"-I{root}", f"--python_out={root}", f"--grpc_python_out={root}",
        str(root / "Common" / "geometry.proto"), str(root / "segmentation.proto"),
    ]
)
if status:
    sys.exit(status)
# geometry.proto has no service: its empty grpc module is not kept.
(root / "Common" / "geometry_pb2_grpc.py").unlink(missing_ok=True)
(root / "Common" / "__init__.py").touch()
for name in ("segmentation_pb2.py", "segmentation_pb2_grpc.py"):
    path = root / name
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^import segmentation_pb2 as", "from . import segmentation_pb2 as", text, flags=re.M)
    text = re.sub(r"^from Common import ", "from .Common import ", text, flags=re.M)
    path.write_text(text, encoding="utf-8")
print("generated", root)
