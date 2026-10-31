"""Block and durably record network/SDK attempts in dedicated test processes.

Loaded at interpreter startup via the test driver's explicit PYTHONPATH. These
are test tripwires for trusted examples, not a sandbox for untrusted code.
"""

import importlib.abc
import json
import os
import sys
from pathlib import Path

ACTIVE = False


def record(kind):
    # Write attempts immediately: multiprocessing workers may exit via os._exit,
    # and upstream code can catch the exception raised by a tripwire.
    path = Path(os.environ["OE_INTEROP_AUDIT_DIR"]) / f"{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, "pid": os.getpid()}) + "\n")


def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
        record(event)
        raise AssertionError("Upstream-example test blocked network access")


class NoCloudImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"alpha_evolve", "google"}:
            record("cloud_import")
            raise AssertionError("Upstream-example test blocked a cloud SDK/auth import")


if os.environ.get("OE_INTEROP_AUDIT_DIR"):
    sys.addaudithook(audit)
    sys.meta_path.insert(0, NoCloudImports())
    record("active")
    ACTIVE = True
