"""Explicit, temporary import aliases and by-value evaluator loading."""

import importlib
import importlib.util
import sys
import threading
import uuid
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Union


@contextmanager
def local_imports() -> Iterator[None]:
    """Temporarily route ``alpha_evolve`` imports to the local facade.

    This opt-in context is process-global and main-thread-only, not thread
    isolation. Use it in a single-threaded entrypoint, not a concurrent service.
    Refuse any pre-existing SDK namespace (including nested local contexts).
    Never change installed packages, import paths, files, or credentials.
    Only this context's own module entries are removed on exit.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Use local_imports on the main thread in a single-threaded entrypoint")
    if any(name == "alpha_evolve" or name.startswith("alpha_evolve.") for name in sys.modules):
        raise RuntimeError("alpha_evolve is already loaded; use a fresh process for local imports")

    modules = {
        name: importlib.import_module(f"openevolve.compat.alpha_evolve.{name}")
        for name in ("client", "controller", "experiment", "models", "visualization")
    }
    package = ModuleType("alpha_evolve", "Explicit, temporary OpenEvolve local facade")
    package.__package__ = "alpha_evolve"
    package.__path__ = []
    package.__spec__ = ModuleSpec("alpha_evolve", loader=None, is_package=True)
    for name, module in modules.items():
        setattr(package, name, module)
    aliases = {"alpha_evolve": package}
    aliases.update({f"alpha_evolve.{name}": module for name, module in modules.items()})
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, module in aliases.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


def load_evaluator(path: Union[str, Path], function_name: str) -> Callable:
    """Import a trusted Python evaluator with local model types, then remove aliases.

    Return the raw candidate-based callable; adapt it with ``adapt_evaluator``
    or pass it to the local ``AlphaEvolveExperiment``. The file module is removed
    after import so cloudpickle transports the callable by value to OE workers.
    Relative/sibling imports and arbitrary import side effects are not managed.
    """
    path = Path(path).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("path must name an existing Python .py evaluator file")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError("function_name must be a Python identifier")
    name = "_openevolve_local_ae_evaluator_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the evaluator file")
    module = importlib.util.module_from_spec(spec)
    with local_imports():
        # Registration during import supports decorators such as @dataclass.
        # Remove it afterwards to select cloudpickle's by-value transport.
        sys.modules[name] = module
        try:
            # Read the requested source, not timestamp/size-based cached bytecode:
            # callers may have just edited a same-size evaluator in this second.
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            evaluator = getattr(module, function_name, None)
            if not callable(evaluator):
                raise ValueError("function_name must select a callable in the evaluator file")
        finally:
            if sys.modules.get(name) is module:
                del sys.modules[name]
    return evaluator
