"""Regenerate seg_proto/segmentation_pb2*.py from seg_proto/segmentation.proto (a copy of Viking's gRPC_Protos file).

Run from annotation-gallery:  python tools/gen_seg_proto.py
The copy of the .proto is the source of truth here; refresh it from Viking's gRPC_Protos/Segmentation/SAM2/segmentation.proto
(commit recorded in seg_proto/PROTO_COMMIT) before regenerating.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from grpc_tools import protoc

root = Path(__file__).resolve().parents[1] / "seg_proto"
status = protoc.main(
    ["protoc", f"-I{root}", f"--python_out={root}", f"--grpc_python_out={root}", str(root / "segmentation.proto")]
)
if status:
    sys.exit(status)
grpc_file = root / "segmentation_pb2_grpc.py"
text = grpc_file.read_text(encoding="utf-8")
text = re.sub(r"^import segmentation_pb2 as", "from . import segmentation_pb2 as", text, flags=re.M)
grpc_file.write_text(text, encoding="utf-8")
print("generated", root)
