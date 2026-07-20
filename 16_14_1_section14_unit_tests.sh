#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Section 14 pre-training unit-test automation for BudgetMem-R.
# Run from the budgetmem-r repository in the VS Code WSL/Bash terminal.
#
# Optional overrides:
#   BUDGETMEM_MODEL_IMPORT="budgetmem.models.budgetmem_r:BudgetMemR"
#   BUDGETMEM_SYNTHETIC_FACTORY="budgetmem.data.selective_copy:SelectiveCopyDataset"
#   SECTION14_STRICT=1
#   INSTALL_DEPS=1
#
# The automation creates:
#   tests/section14_runtime.py
#   tests/test_section14_required.py
#   reports/evidence/section14_unit_tests_report.txt
#   reports/evidence/logs/section14_unit_tests_<timestamp>.log
#   reports/evidence/junit/section14_unit_tests_<timestamp>.xml
#   reports/tables/section14_unit_test_results.csv

readonly SCRIPT_NAME="$(basename "$0")"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly SECTION14_STRICT="${SECTION14_STRICT:-1}"
readonly INSTALL_DEPS="${INSTALL_DEPS:-1}"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

find_repo_root() {
    if root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        printf '%s\n' "$root"
        return 0
    fi

    local cursor
    cursor="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "$cursor" != "/" ]]; do
        if [[ -f "$cursor/pyproject.toml" && -d "$cursor/src" ]]; then
            printf '%s\n' "$cursor"
            return 0
        fi
        cursor="$(dirname "$cursor")"
    done

    return 1
}

REPO_ROOT="$(find_repo_root)" || die "Repository root not found. Save this file inside budgetmem-r and run it from the VS Code Bash/WSL terminal."
cd "$REPO_ROOT"

[[ -d src/budgetmem ]] || die "Expected package directory not found: $REPO_ROOT/src/budgetmem"
[[ -f pyproject.toml ]] || die "Expected pyproject.toml not found in $REPO_ROOT"

choose_python() {
    local candidate
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/venv/bin/python" \
        "$REPO_ROOT/.env/bin/python" \
        python3 \
        python
    do
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(choose_python)" || die "Python was not found. Activate the project virtual environment first."
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SECTION14_STRICT

mkdir -p \
    tests \
    reports/evidence/logs \
    reports/evidence/junit \
    reports/tables

backup_if_present() {
    local path="$1"
    if [[ -f "$path" ]]; then
        cp "$path" "${path}.backup_${TIMESTAMP}"
    fi
}

backup_if_present tests/section14_runtime.py
backup_if_present tests/test_section14_required.py

log "Repository: $REPO_ROOT"
log "Python: $PYTHON_BIN"
log "Creating the Section 14 runtime adapter and strict tests."

cat > tests/section14_runtime.py <<'PY'
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import random
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STRICT = os.getenv("SECTION14_STRICT", "1") == "1"

MODEL_EXACT_NAMES = (
    "BudgetMemR",
    "BudgetMemRModel",
    "BudgetMemoryRNN",
    "BudgetedMemoryRNN",
)
MODEL_MODULE_HINTS = (
    "budgetmem.models.budgetmem_r",
    "budgetmem.models.budgetmem",
    "budgetmem.models.memory_rnn",
    "budgetmem.model",
)
SYNTHETIC_TASK_HINTS = (
    "selective_copy",
    "associative_recall",
    "distractor_retrieval",
)
CONTROLLER_WORDS = (
    "controller",
    "write",
    "retain",
    "retention",
    "utility",
    "policy",
    "selector",
    "evict",
)
MEMORY_WORDS = (
    "memory",
    "cache",
    "slot",
    "external",
    "bank",
)


class Section14DiscoveryError(RuntimeError):
    pass


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.getenv("SECTION14_TORCH_THREADS", "1"))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _iter_budgetmem_module_names() -> list[str]:
    names = list(MODEL_MODULE_HINTS)
    package_root = SRC / "budgetmem"
    if package_root.exists():
        for path in package_root.rglob("*.py"):
            if path.name == "__init__.py":
                module = ".".join(path.parent.relative_to(SRC).parts)
            else:
                module = ".".join(path.with_suffix("").relative_to(SRC).parts)
            if module not in names:
                names.append(module)
    return names


def _safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _resolve_override(value: str) -> Any:
    if ":" not in value:
        raise Section14DiscoveryError(
            f"Override must use module:object syntax, received {value!r}."
        )
    module_name, object_name = value.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def discover_model_class() -> type[nn.Module]:
    override = os.getenv("BUDGETMEM_MODEL_IMPORT")
    if override:
        candidate = _resolve_override(override)
        if not inspect.isclass(candidate) or not issubclass(candidate, nn.Module):
            raise Section14DiscoveryError(
                f"BUDGETMEM_MODEL_IMPORT={override!r} does not resolve to torch.nn.Module."
            )
        return candidate

    candidates: list[tuple[int, type[nn.Module]]] = []
    for module_name in _iter_budgetmem_module_names():
        module = _safe_import(module_name)
        if module is None:
            continue
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not issubclass(obj, nn.Module):
                continue
            if obj is nn.Module or obj.__module__ != module.__name__:
                continue
            lowered = name.lower()
            module_lowered = module_name.lower()
            score = 0
            if name in MODEL_EXACT_NAMES:
                score += 100
            if "budget" in lowered and "mem" in lowered:
                score += 60
            if "budgetmem" in module_lowered:
                score += 35
            if any(word in lowered for word in ("gru", "lstm", "rnn")):
                score += 5
            if any(word in lowered for word in ("controller", "memory", "cache")):
                score -= 25
            if score > 0:
                candidates.append((score, obj))

    if not candidates:
        raise Section14DiscoveryError(
            "BudgetMem-R model class was not discovered. Set "
            "BUDGETMEM_MODEL_IMPORT='module.path:ClassName'."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _annotation_class(annotation: Any) -> type[Any] | None:
    if inspect.isclass(annotation):
        return annotation
    return None


def _default_value(name: str, *, budget: int, force_detach: bool | None) -> Any:
    key = name.lower()
    if key in {"budget", "memory_budget", "max_memory", "max_memory_size", "capacity", "num_slots", "memory_size"}:
        return budget
    if key in {"input_size", "input_dim", "feature_dim", "embedding_dim", "embed_dim", "d_input"}:
        return 8
    if key in {"hidden_size", "hidden_dim", "d_model", "model_dim", "controller_hidden_size"}:
        return 16
    if key in {"output_size", "output_dim", "num_classes", "n_classes", "vocab_size", "num_embeddings"}:
        return 32
    if key in {"key_dim", "memory_key_dim"}:
        return 8
    if key in {"value_dim", "memory_value_dim"}:
        return 16
    if key in {"num_layers", "n_layers", "layers", "heads", "num_heads"}:
        return 1
    if key in {"dropout", "dropout_rate"}:
        return 0.0
    if key in {"batch_first"}:
        return True
    if key in {"device"}:
        return "cpu"
    if key in {"dtype"}:
        return torch.float32
    if key in {"task", "task_name"}:
        return "selective_copy"
    if key in {"seed", "random_seed"}:
        return 2026
    if any(token in key for token in ("detach", "truncate_graph", "stop_gradient")):
        return True if force_detach is None else force_detach
    if key in {"trainable_memory", "trainable_cache", "retain_graph_memory"}:
        return False if force_detach is None else not force_detach
    return _MISSING


class _Missing:
    pass


_MISSING = _Missing()


def _build_dataclass_config(
    config_cls: type[Any], *, budget: int, force_detach: bool | None
) -> Any:
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(config_cls):
        if field.default is not dataclasses.MISSING:
            continue
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            continue
        value = _default_value(field.name, budget=budget, force_detach=force_detach)
        if value is _MISSING:
            raise Section14DiscoveryError(
                f"Cannot infer required config field {config_cls.__name__}.{field.name}."
            )
        kwargs[field.name] = value
    config = config_cls(**kwargs)
    for field in dataclasses.fields(config):
        value = _default_value(field.name, budget=budget, force_detach=force_detach)
        if value is not _MISSING:
            try:
                setattr(config, field.name, value)
            except Exception:
                pass
    return config


def _find_config_class(model_cls: type[nn.Module]) -> type[Any] | None:
    module = importlib.import_module(model_cls.__module__)
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if "config" not in name.lower():
            continue
        if dataclasses.is_dataclass(obj):
            return obj
    return None


def build_model(
    *,
    seed: int = 2026,
    budget: int = 4,
    force_detach: bool | None = None,
) -> nn.Module:
    set_all_seeds(seed)
    model_cls = discover_model_class()
    signature = inspect.signature(model_cls)
    kwargs: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        value = _default_value(name, budget=budget, force_detach=force_detach)
        if value is not _MISSING:
            kwargs[name] = value
            continue

        if name.lower() in {"config", "cfg"}:
            config_cls = _annotation_class(parameter.annotation) or _find_config_class(model_cls)
            if config_cls is not None and dataclasses.is_dataclass(config_cls):
                kwargs[name] = _build_dataclass_config(
                    config_cls,
                    budget=budget,
                    force_detach=force_detach,
                )
                continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        raise Section14DiscoveryError(
            f"Cannot instantiate {model_cls.__module__}:{model_cls.__name__}; "
            f"required constructor parameter {name!r} is unknown. "
            "Set BUDGETMEM_MODEL_IMPORT or extend the generated adapter."
        )

    model = model_cls(**kwargs)
    model.to("cpu")

    for object_ in _walk_named_objects(model):
        for attr in (
            "budget",
            "memory_budget",
            "max_memory",
            "max_memory_size",
            "capacity",
            "num_slots",
        ):
            if hasattr(object_, attr):
                try:
                    current = getattr(object_, attr)
                    if isinstance(current, (int, float)):
                        setattr(object_, attr, budget)
                except Exception:
                    pass

        if force_detach is not None:
            for attr in (
                "detach_cached_states",
                "detach_cache",
                "detach_memory",
                "truncate_memory_graph",
            ):
                if hasattr(object_, attr):
                    try:
                        setattr(object_, attr, force_detach)
                    except Exception:
                        pass
            for attr in ("trainable_memory", "trainable_cache"):
                if hasattr(object_, attr):
                    try:
                        setattr(object_, attr, not force_detach)
                    except Exception:
                        pass

    return model


def _walk_named_objects(model: nn.Module) -> Iterable[Any]:
    yielded: set[int] = set()
    for object_ in [model, *list(model.modules())]:
        if id(object_) not in yielded:
            yielded.add(id(object_))
            yield object_
        for name in MEMORY_WORDS:
            if hasattr(object_, name):
                child = getattr(object_, name)
                if child is not None and id(child) not in yielded:
                    yielded.add(id(child))
                    yield child


def _forward_parameters(model: nn.Module) -> list[inspect.Parameter]:
    return [
        parameter
        for name, parameter in inspect.signature(model.forward).parameters.items()
        if name != "self"
    ]


def _candidate_inputs(model: nn.Module, seq_len: int = 12) -> list[Tensor]:
    batch = 2
    vocab = 32
    features = 8
    for object_ in _walk_named_objects(model):
        for attr in ("vocab_size", "num_embeddings", "input_vocab_size"):
            value = getattr(object_, attr, None)
            if isinstance(value, int) and value > 2:
                vocab = value
        for attr in ("input_size", "input_dim", "feature_dim", "embedding_dim"):
            value = getattr(object_, attr, None)
            if isinstance(value, int) and value > 0:
                features = value

    integer_batch_first = torch.randint(0, max(vocab, 3), (batch, seq_len), dtype=torch.long)
    integer_time_first = integer_batch_first.transpose(0, 1).contiguous()
    float_batch_first = torch.randn(batch, seq_len, features)
    float_time_first = float_batch_first.transpose(0, 1).contiguous()
    one_hot = torch.nn.functional.one_hot(
        integer_batch_first % features, num_classes=features
    ).float()

    return [
        integer_batch_first,
        float_batch_first,
        one_hot,
        integer_time_first,
        float_time_first,
    ]


def _forward_kwargs(model: nn.Module, x: Tensor, *, reset: bool | None) -> dict[str, Any]:
    parameters = _forward_parameters(model)
    kwargs: dict[str, Any] = {}
    for parameter in parameters[1:]:
        name = parameter.name
        key = name.lower()
        if key in {"budget", "memory_budget"}:
            kwargs[name] = 4
        elif key in {"reset", "reset_memory", "clear_memory"} and reset is not None:
            kwargs[name] = reset
        elif key in {"return_diagnostics", "return_memory", "return_state", "return_details"}:
            kwargs[name] = True
        elif key in {"lengths", "sequence_lengths"}:
            seq_len = x.shape[1] if x.ndim >= 2 else x.shape[0]
            batch = x.shape[0] if x.ndim >= 2 else 1
            kwargs[name] = torch.full((batch,), seq_len, dtype=torch.long)
        elif key in {"mask", "attention_mask", "padding_mask"}:
            kwargs[name] = torch.ones(x.shape[:2], dtype=torch.bool)
        elif key in {"hidden", "hidden_state", "hx", "state"}:
            kwargs[name] = None
        elif key in {"targets", "target", "labels", "y"}:
            kwargs[name] = x.clone()
        elif parameter.default is inspect.Parameter.empty:
            raise Section14DiscoveryError(
                f"Cannot infer required forward parameter {name!r} for "
                f"{type(model).__module__}:{type(model).__name__}."
            )
    return kwargs


def invoke(model: nn.Module, x: Tensor, *, reset: bool | None = None) -> Any:
    parameters = _forward_parameters(model)
    if not parameters:
        raise Section14DiscoveryError("Model.forward has no input parameter.")
    kwargs = _forward_kwargs(model, x, reset=reset)
    return model(x, **kwargs)


def compatible_input(model: nn.Module, seq_len: int = 12) -> Tensor:
    failures: list[str] = []
    for x in _candidate_inputs(model, seq_len=seq_len):
        try:
            reset_memory(model, require=False)
            with torch.no_grad():
                invoke(model, x, reset=True)
            return x
        except Exception as exc:
            failures.append(f"shape={tuple(x.shape)}, dtype={x.dtype}: {type(exc).__name__}: {exc}")
    raise Section14DiscoveryError(
        "No compatible synthetic input was found for the discovered model. Attempts:\n"
        + "\n".join(failures)
    )


def extract_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            tensor = extract_tensor(getattr(value, field.name))
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        preferred = ("logits", "output", "outputs", "prediction", "predictions", "hidden")
        for key in preferred:
            if key in value:
                tensor = extract_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def infer_sequence_axis(tensor: Tensor, seq_len: int) -> int | None:
    matching = [index for index, size in enumerate(tensor.shape) if size == seq_len]
    if not matching:
        return None
    if tensor.ndim >= 3 and 1 in matching:
        return 1
    return matching[0]


def sequence_prefix(tensor: Tensor, seq_len: int, prefix_len: int) -> Tensor | None:
    axis = infer_sequence_axis(tensor, seq_len)
    if axis is None:
        return None
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(0, prefix_len)
    return tensor[tuple(slices)]


def input_sequence_axis(x: Tensor) -> int:
    if x.ndim == 1:
        return 0
    if x.ndim == 2:
        return 1 if x.shape[0] <= x.shape[1] else 0
    if x.ndim >= 3:
        return 1 if x.shape[0] <= x.shape[1] else 0
    raise Section14DiscoveryError(f"Unsupported input rank: {x.ndim}")


def slice_step(x: Tensor, step: int) -> Tensor:
    axis = input_sequence_axis(x)
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(step, step + 1)
    return x[tuple(slices)]


def mutate_suffix(x: Tensor, prefix_len: int) -> Tensor:
    changed = x.clone()
    axis = input_sequence_axis(changed)
    slices = [slice(None)] * changed.ndim
    slices[axis] = slice(prefix_len, None)
    suffix = changed[tuple(slices)]
    if changed.dtype.is_floating_point:
        changed[tuple(slices)] = suffix + 7.0
    else:
        max_value = 31
        changed[tuple(slices)] = (suffix + 11) % max_value
    return changed


def reset_memory(model: nn.Module, *, require: bool = True) -> bool:
    method_names = (
        "reset_memory",
        "clear_memory",
        "reset_state",
        "clear_state",
        "reset",
    )
    for object_ in _walk_named_objects(model):
        for name in method_names:
            method = getattr(object_, name, None)
            if callable(method):
                try:
                    signature = inspect.signature(method)
                    required = [
                        p
                        for p in signature.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind
                        not in {
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                        }
                    ]
                    if not required:
                        method()
                        return True
                except (TypeError, ValueError):
                    try:
                        method()
                        return True
                    except Exception:
                        pass
    if require:
        raise Section14DiscoveryError(
            "No zero-argument memory reset method was discovered. "
            "Provide reset_memory(), clear_memory(), reset_state(), or clear_state()."
        )
    return False


def encourage_writes(model: nn.Module) -> None:
    for object_ in _walk_named_objects(model):
        for attr in (
            "write_threshold",
            "retention_threshold",
            "admission_threshold",
            "store_threshold",
        ):
            if hasattr(object_, attr):
                try:
                    setattr(object_, attr, -1.0)
                except Exception:
                    pass
        for attr in ("force_write", "always_write"):
            if hasattr(object_, attr):
                try:
                    setattr(object_, attr, True)
                except Exception:
                    pass


def _numeric_size(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Tensor) and value.numel() == 1:
        return int(value.detach().cpu().item())
    return None


def memory_size(model: nn.Module) -> int | None:
    sizes: list[int] = []
    for object_ in _walk_named_objects(model):
        for attr in (
            "memory_size",
            "current_size",
            "num_entries",
            "num_slots_used",
            "n_items",
            "size",
        ):
            if not hasattr(object_, attr):
                continue
            try:
                value = getattr(object_, attr)
                if callable(value):
                    value = value()
                number = _numeric_size(value)
                if number is not None and number >= 0:
                    sizes.append(number)
            except Exception:
                pass

        for attr in ("slots", "entries", "items", "memory", "cache"):
            if not hasattr(object_, attr):
                continue
            try:
                value = getattr(object_, attr)
                if isinstance(value, (list, tuple, dict)):
                    sizes.append(len(value))
            except Exception:
                pass

        tensor_candidates: list[Tensor] = []
        for attr in ("keys", "values", "memory_keys", "memory_values", "valid_mask", "occupied"):
            value = getattr(object_, attr, None)
            if isinstance(value, Tensor) and value.ndim > 0:
                tensor_candidates.append(value)
        for tensor in tensor_candidates:
            if tensor.dtype == torch.bool:
                sizes.append(int(tensor.detach().sum().cpu().item()))
            elif tensor.ndim == 1:
                sizes.append(int(tensor.shape[0]))
            elif tensor.ndim >= 2:
                dimensions = [dim for dim in tensor.shape if 0 <= dim <= 4096]
                if dimensions:
                    sizes.append(int(min(dimensions)))

    sensible = [size for size in sizes if 0 <= size <= 1_000_000]
    return max(sensible) if sensible else None


def memory_tensors(model: nn.Module) -> list[Tensor]:
    tensors: list[Tensor] = []
    seen: set[int] = set()
    for object_ in _walk_named_objects(model):
        if isinstance(object_, Tensor) and id(object_) not in seen:
            seen.add(id(object_))
            tensors.append(object_)
        if hasattr(object_, "__dict__"):
            for name, value in vars(object_).items():
                tensor_terms = MEMORY_WORDS + CONTROLLER_WORDS + (
                    "key",
                    "value",
                    "state",
                    "entry",
                    "item",
                    "hidden",
                )
                if not any(word in name.lower() for word in tensor_terms):
                    continue
                if isinstance(value, Tensor) and id(value) not in seen:
                    seen.add(id(value))
                    tensors.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, Tensor) and id(item) not in seen:
                            seen.add(id(item))
                            tensors.append(item)
    return tensors


def controller_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(word in name.lower() for word in CONTROLLER_WORDS)
    ]
    if selected:
        return selected

    selected_modules = [
        (name, module)
        for name, module in model.named_modules()
        if any(word in name.lower() for word in CONTROLLER_WORDS)
    ]
    for module_name, module in selected_modules:
        for name, parameter in module.named_parameters(recurse=True):
            selected.append((f"{module_name}.{name}".strip("."), parameter))
    return selected


def capture_controller_outputs(model: nn.Module, x: Tensor) -> tuple[Any, list[Tensor]]:
    captured: list[Tensor] = []
    handles: list[Any] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = extract_tensor(output)
        if tensor is not None:
            captured.append(tensor.detach().cpu().clone())

    for name, module in model.named_modules():
        if name and any(word in name.lower() for word in CONTROLLER_WORDS):
            handles.append(module.register_forward_hook(hook))

    try:
        result = invoke(model, x, reset=True)
    finally:
        for handle in handles:
            handle.remove()
    return result, captured


def state_dict_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(
        torch.equal(left_state[key].detach().cpu(), right_state[key].detach().cpu())
        for key in left_state
    )


def output_equal(left: Any, right: Any) -> bool:
    left_tensor = extract_tensor(left)
    right_tensor = extract_tensor(right)
    if left_tensor is None or right_tensor is None:
        return repr(left) == repr(right)
    return torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu())


def _callable_required_kwargs(
    callable_: Callable[..., Any],
    *,
    seed: int,
    split: str = "train",
) -> dict[str, Any]:
    signature = inspect.signature(callable_)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls", "args", "kwargs"}:
            continue
        key = name.lower()
        value: Any = _MISSING
        if key in {"seed", "random_seed", "rng_seed"}:
            value = seed
        elif key in {"split", "partition"}:
            value = split
        elif key in {"num_samples", "n_samples", "size", "dataset_size", "examples"}:
            value = 16
        elif key in {"sequence_length", "seq_len", "length", "max_length"}:
            value = 32
        elif key in {"vocab_size", "num_tokens", "alphabet_size"}:
            value = 16
        elif key in {"memory_budget", "budget"}:
            value = 4
        elif key in {"root", "data_dir", "cache_dir", "path"}:
            value = str(ROOT / "data")
        elif key in {"download"}:
            value = False
        if value is not _MISSING:
            kwargs[name] = value
        elif parameter.default is inspect.Parameter.empty:
            raise Section14DiscoveryError(
                f"Cannot infer required dataset-factory parameter {name!r} "
                f"for {callable_!r}."
            )
    return kwargs


def discover_synthetic_factory() -> Callable[..., Any]:
    override = os.getenv("BUDGETMEM_SYNTHETIC_FACTORY")
    if override:
        candidate = _resolve_override(override)
        if not callable(candidate):
            raise Section14DiscoveryError(
                f"BUDGETMEM_SYNTHETIC_FACTORY={override!r} is not callable."
            )
        return candidate

    candidates: list[tuple[int, Callable[..., Any]]] = []
    for module_name in _iter_budgetmem_module_names():
        if not any(hint in module_name.lower() for hint in SYNTHETIC_TASK_HINTS + ("data", "task", "dataset")):
            continue
        module = _safe_import(module_name)
        if module is None:
            continue
        for name, obj in vars(module).items():
            if not callable(obj):
                continue
            lowered = name.lower()
            score = 0
            if any(hint in lowered for hint in SYNTHETIC_TASK_HINTS):
                score += 50
            if any(word in lowered for word in ("dataset", "generate", "build", "create", "make")):
                score += 25
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, torch.utils.data.Dataset):
                        score += 40
                except TypeError:
                    pass
            if score > 0:
                candidates.append((score, obj))

    if not candidates:
        raise Section14DiscoveryError(
            "Synthetic task factory was not discovered. Set "
            "BUDGETMEM_SYNTHETIC_FACTORY='module.path:factory'."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_synthetic_dataset(seed: int, split: str = "train") -> Any:
    factory = discover_synthetic_factory()
    kwargs = _callable_required_kwargs(factory, seed=seed, split=split)
    return factory(**kwargs)


def _stable_serialize(value: Any) -> bytes:
    if isinstance(value, Tensor):
        array = value.detach().cpu().contiguous().numpy()
        return b"TENSOR|" + str(array.dtype).encode() + b"|" + str(array.shape).encode() + b"|" + array.tobytes()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return b"NDARRAY|" + str(array.dtype).encode() + b"|" + str(array.shape).encode() + b"|" + array.tobytes()
    if dataclasses.is_dataclass(value):
        return _stable_serialize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        chunks = []
        for key in sorted(value, key=lambda item: str(item)):
            chunks.append(_stable_serialize(key))
            chunks.append(_stable_serialize(value[key]))
        return b"MAP|" + b"|".join(chunks)
    if isinstance(value, (list, tuple)):
        return b"SEQ|" + b"|".join(_stable_serialize(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value).encode("utf-8")
    return repr(value).encode("utf-8")


def dataset_fingerprint(dataset: Any, limit: int = 16) -> str:
    digest = hashlib.sha256()
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        count = min(int(len(dataset)), limit)
        for index in range(count):
            digest.update(_stable_serialize(dataset[index]))
    else:
        for index, item in enumerate(dataset):
            if index >= limit:
                break
            digest.update(_stable_serialize(item))
    return digest.hexdigest()


def project_text_files() -> Iterable[Path]:
    for base in (ROOT / "src", ROOT / "tests", ROOT / "configs", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml"}:
                yield path


def existing_test_has(*groups: Sequence[str]) -> bool:
    for path in (ROOT / "tests").rglob("test*.py"):
        if path.name == "test_section14_required.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if all(any(term.lower() in text for term in group) for group in groups):
            return True
    return False


def source_has_deterministic_loader_controls() -> bool:
    terms = (
        "worker_init_fn",
        "torch.generator",
        "manual_seed",
        "use_deterministic_algorithms",
        "deterministic",
    )
    for path in project_text_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "dataloader" in text and any(term in text for term in terms):
            return True
    return existing_test_has(("training", "order"), ("determin", "seed"))


def _walk_config(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_config(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_config(item, (*path, str(index)))
    else:
        yield path, value


def explicit_split_seeds() -> dict[str, set[int]]:
    result: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    config_paths = list((ROOT / "configs").rglob("*.yaml")) + list((ROOT / "configs").rglob("*.yml")) + list((ROOT / "configs").rglob("*.json"))
    for path in config_paths:
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                import yaml

                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key_path, value in _walk_config(payload):
            lowered = ".".join(key_path).lower()
            if "seed" not in lowered:
                continue
            if not isinstance(value, int):
                continue
            if "train" in lowered:
                result["train"].add(value)
            if "validation" in lowered or re.search(r"(^|[._-])val([._-]|$)", lowered):
                result["validation"].add(value)
            if "test" in lowered:
                result["test"].add(value)
    return result


def _read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if path.stat().st_size > 200 * 1024 * 1024:
        return []
    try:
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        rows.append(dict(value))
            return rows
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
            if isinstance(value, Mapping):
                for key in ("records", "data", "examples", "items"):
                    rows = value.get(key)
                    if isinstance(rows, list):
                        return [dict(item) for item in rows if isinstance(item, Mapping)]
            return []
        if suffix in {".csv", ".tsv"}:
            import csv

            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
        if suffix == ".parquet":
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
    except Exception:
        return []
    return []


def _infer_split(path: Path, row: Mapping[str, Any]) -> str | None:
    for key in ("split", "partition", "subset", "fold"):
        value = row.get(key)
        if value is not None:
            lowered = str(value).lower()
            if lowered in {"train", "training"}:
                return "train"
            if lowered in {"val", "valid", "validation", "dev"}:
                return "validation"
            if lowered in {"test", "testing"}:
                return "test"
    lowered_name = str(path).lower()
    if "train" in lowered_name:
        return "train"
    if any(token in lowered_name for token in ("validation", "_val", "-val", "/val", "dev")):
        return "validation"
    if "test" in lowered_name:
        return "test"
    return None


def _record_identity(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for field in fields:
        if field in lowered and lowered[field] not in {None, ""}:
            normalized = re.sub(r"\s+", " ", str(lowered[field]).strip())
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return None


def local_split_identities(kind: str, identity_fields: Sequence[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    data_root = ROOT / "data"
    if not data_root.exists():
        return result
    supported = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        if kind.lower() not in str(path).lower():
            continue
        for row in _read_records(path):
            split = _infer_split(path, row)
            identity = _record_identity(row, identity_fields)
            if split is not None and identity is not None:
                result[split].add(identity)
    return result


def cache_graph_policy_evidence() -> dict[str, bool]:
    evidence = {
        "detach_call": False,
        "explicit_policy": False,
        "trainable_path": False,
    }
    for path in (ROOT / "src" / "budgetmem").rglob("*.py"):
        lowered_path = str(path).lower()
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        relevant_terms = MEMORY_WORDS + CONTROLLER_WORDS + ("budgetmem", "cached")
        if not any(word in lowered_path or word in text for word in relevant_terms):
            continue
        if ".detach(" in text or ".detach()" in text:
            evidence["detach_call"] = True
        if any(
            token in text
            for token in (
                "detach_cached_states",
                "detach_cache",
                "detach_memory",
                "truncate_memory_graph",
                "intentional detach",
                "intentionally detached",
            )
        ):
            evidence["explicit_policy"] = True
        if any(
            token in text
            for token in (
                "trainable_cached_states",
                "trainable_cache",
                "trainable_memory",
                "retain_graph",
                "detach_cached_states = false",
                "detach_cache = false",
            )
        ):
            evidence["trainable_path"] = True
    return evidence


def supports_detach_override(model: nn.Module) -> bool:
    names: set[str] = set(inspect.signature(type(model)).parameters)
    for object_ in _walk_named_objects(model):
        if hasattr(object_, "__dict__"):
            names.update(vars(object_))
    lowered = {name.lower() for name in names}
    return any("detach" in name or "trainable_memory" in name or "trainable_cache" in name for name in lowered)
PY

cat > tests/test_section14_required.py <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

# Pytest can use importlib collection mode, in which the tests directory is not
# automatically importable as a top-level module. Add it explicitly so the
# generated runtime adapter can always be imported.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from section14_runtime import (
    Section14DiscoveryError,
    build_model,
    build_synthetic_dataset,
    cache_graph_policy_evidence,
    capture_controller_outputs,
    compatible_input,
    controller_parameters,
    dataset_fingerprint,
    encourage_writes,
    existing_test_has,
    explicit_split_seeds,
    extract_tensor,
    infer_sequence_axis,
    input_sequence_axis,
    invoke,
    local_split_identities,
    memory_size,
    memory_tensors,
    mutate_suffix,
    output_equal,
    reset_memory,
    sequence_prefix,
    set_all_seeds,
    slice_step,
    source_has_deterministic_loader_controls,
    state_dict_equal,
    supports_detach_override,
)


def _sequence_length(x: torch.Tensor) -> int:
    return int(x.shape[input_sequence_axis(x)])


def _assert_no_overlap(
    left_name: str,
    left: set[object],
    right_name: str,
    right: set[object],
) -> None:
    overlap = left & right
    assert not overlap, (
        f"{left_name} and {right_name} overlap. "
        f"Overlap count={len(overlap)}; sample={list(overlap)[:5]}"
    )


def test_14_01_budget_correctness_every_forward_step() -> None:
    budget = 4
    model = build_model(seed=2026, budget=budget)
    model.eval()
    encourage_writes(model)
    x = compatible_input(model, seq_len=12)
    reset_memory(model)
    observed: list[int] = []

    with torch.no_grad():
        for step in range(_sequence_length(x)):
            invoke(model, slice_step(x, step), reset=False)
            size = memory_size(model)
            assert size is not None, (
                "The runtime adapter could not inspect memory size. Expose one of "
                "memory_size/current_size/num_entries/num_slots_used or a keys/values tensor."
            )
            observed.append(size)
            assert size <= budget, (
                f"Memory budget violated at forward step {step}: "
                f"memory.size={size}, configured_budget={budget}."
            )

    assert observed, "No forward steps were checked."


def test_14_02_causality_future_tokens_do_not_change_earlier_steps() -> None:
    seed = 2026
    model_a = build_model(seed=seed, budget=4)
    model_b = build_model(seed=seed, budget=4)
    assert state_dict_equal(model_a, model_b), "Models are not identically initialized."

    model_a.eval()
    model_b.eval()
    x = compatible_input(model_a, seq_len=12)
    seq_len = _sequence_length(x)
    prefix_len = max(2, seq_len // 2)
    changed = mutate_suffix(x, prefix_len)

    reset_memory(model_a, require=False)
    output_a, decisions_a = capture_controller_outputs(model_a, x)
    reset_memory(model_b, require=False)
    output_b, decisions_b = capture_controller_outputs(model_b, changed)

    tensor_a = extract_tensor(output_a)
    tensor_b = extract_tensor(output_b)
    compared = False

    if tensor_a is not None and tensor_b is not None:
        prefix_a = sequence_prefix(tensor_a, seq_len, prefix_len)
        prefix_b = sequence_prefix(tensor_b, seq_len, prefix_len)
        if prefix_a is not None and prefix_b is not None:
            assert torch.equal(prefix_a.cpu(), prefix_b.cpu()), (
                "Changing future tokens changed model outputs at earlier time steps."
            )
            compared = True

    for left, right in zip(decisions_a, decisions_b, strict=False):
        axis_left = infer_sequence_axis(left, seq_len)
        axis_right = infer_sequence_axis(right, seq_len)
        if axis_left is None or axis_right is None:
            continue
        prefix_left = sequence_prefix(left, seq_len, prefix_len)
        prefix_right = sequence_prefix(right, seq_len, prefix_len)
        if prefix_left is not None and prefix_right is not None:
            assert torch.equal(prefix_left, prefix_right), (
                "Changing future tokens changed controller decisions at earlier steps."
            )
            compared = True

    assert compared, (
        "Causality could not be evaluated because neither sequence-aligned model outputs "
        "nor sequence-aligned controller decisions were exposed."
    )


def test_14_03_deterministic_dataset_generation() -> None:
    try:
        first = build_synthetic_dataset(seed=2026, split="train")
        second = build_synthetic_dataset(seed=2026, split="train")
    except Section14DiscoveryError:
        assert existing_test_has(
            ("synthetic", "dataset"),
            ("determin", "same seed", "reproduc"),
        ), (
            "Synthetic dataset factory was not discovered and no existing deterministic "
            "synthetic-dataset test was found."
        )
        return

    assert dataset_fingerprint(first) == dataset_fingerprint(second), (
        "The same synthetic seed and configuration produced different datasets."
    )


def test_14_04_deterministic_initialization() -> None:
    first = build_model(seed=2026, budget=4)
    second = build_model(seed=2026, budget=4)
    assert state_dict_equal(first, second), (
        "The same seed and configuration produced different model initialization."
    )


def test_14_05_deterministic_training_order() -> None:
    assert source_has_deterministic_loader_controls(), (
        "Training code does not expose deterministic DataLoader controls. "
        "Use a seeded torch.Generator and deterministic worker_init_fn."
    )

    dataset = TensorDataset(torch.arange(32))
    generator_a = torch.Generator().manual_seed(2026)
    generator_b = torch.Generator().manual_seed(2026)
    loader_a = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator_a)
    loader_b = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator_b)
    order_a = torch.cat([batch[0] for batch in loader_a])
    order_b = torch.cat([batch[0] for batch in loader_b])
    assert torch.equal(order_a, order_b), (
        "The same DataLoader seed produced different training order."
    )


def test_14_06_deterministic_evaluation_output() -> None:
    first = build_model(seed=2026, budget=4)
    second = build_model(seed=2026, budget=4)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    first.eval()
    second.eval()
    x = compatible_input(first, seq_len=10)

    reset_memory(first, require=False)
    reset_memory(second, require=False)
    with torch.no_grad():
        output_a = invoke(first, x, reset=True)
        output_b = invoke(second, x.clone(), reset=True)

    assert output_equal(output_a, output_b), (
        "Identical seed, configuration, state, and evaluation input produced different output. "
        "CUDA nondeterminism must be explicitly documented; this CPU gate must remain exact."
    )


def test_14_07_synthetic_seeds_do_not_overlap() -> None:
    seeds = explicit_split_seeds()
    if all(seeds.values()):
        _assert_no_overlap("train seeds", seeds["train"], "validation seeds", seeds["validation"])
        _assert_no_overlap("train seeds", seeds["train"], "test seeds", seeds["test"])
        _assert_no_overlap("validation seeds", seeds["validation"], "test seeds", seeds["test"])
        return

    assert existing_test_has(
        ("synthetic", "seed"),
        ("train", "validation", "test"),
        ("overlap", "disjoint", "leak"),
    ), (
        "Explicit train/validation/test synthetic seeds were not found in configs, and no "
        "existing split-seed leakage test was found."
    )


def test_14_08_hdfs_block_ids_do_not_overlap() -> None:
    identities = local_split_identities(
        "hdfs",
        ("block_id", "blockid", "block", "id"),
    )
    if all(identities.values()):
        _assert_no_overlap("HDFS train block IDs", identities["train"], "HDFS validation block IDs", identities["validation"])
        _assert_no_overlap("HDFS train block IDs", identities["train"], "HDFS test block IDs", identities["test"])
        _assert_no_overlap("HDFS validation block IDs", identities["validation"], "HDFS test block IDs", identities["test"])
        return

    assert existing_test_has(
        ("hdfs",),
        ("block", "block_id"),
        ("overlap", "disjoint", "leak"),
    ), (
        "Prepared HDFS split records were not discovered under data/, and no existing "
        "HDFS block-ID leakage test was found."
    )


def test_14_09_imdb_test_examples_not_in_train_or_validation() -> None:
    identities = local_split_identities(
        "imdb",
        ("text", "review", "content", "sentence", "example_id", "id"),
    )
    if identities["test"] and (identities["train"] or identities["validation"]):
        _assert_no_overlap("IMDb train examples", identities["train"], "IMDb official test examples", identities["test"])
        _assert_no_overlap("IMDb validation examples", identities["validation"], "IMDb official test examples", identities["test"])
        return

    assert existing_test_has(
        ("imdb",),
        ("official", "test"),
        ("train", "validation"),
        ("overlap", "included", "leak", "disjoint"),
    ), (
        "Prepared IMDb split records were not discovered under data/, and no existing "
        "test proving official-test isolation was found."
    )


def test_14_10_memory_controller_parameters_receive_gradients() -> None:
    model = build_model(seed=2026, budget=4, force_detach=False)
    model.train()
    encourage_writes(model)
    x = compatible_input(model, seq_len=10)
    reset_memory(model, require=False)
    model.zero_grad(set_to_none=True)
    output = invoke(model, x, reset=True)
    tensor = extract_tensor(output)
    assert tensor is not None, "No differentiable tensor was found in model output."
    assert tensor.requires_grad, "Model output is detached; gradient-flow test cannot proceed."

    loss = tensor.float().pow(2).mean()
    loss.backward()

    parameters = controller_parameters(model)
    assert parameters, (
        "No controller parameters were discovered. Controller parameter names or module "
        "names must include controller/write/retention/utility/policy/selector."
    )
    missing = [name for name, parameter in parameters if parameter.requires_grad and parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in parameters
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    nonzero = [
        name
        for name, parameter in parameters
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
    ]
    assert not missing, f"Controller parameters did not receive gradients: {missing}"
    assert not nonfinite, f"Controller parameters received non-finite gradients: {nonfinite}"
    assert nonzero, "All controller gradients are zero."


def test_14_11_detached_cached_states_are_intentionally_detached() -> None:
    evidence = cache_graph_policy_evidence()
    assert evidence["detach_call"], (
        "No explicit cached-state detach operation was found in memory/controller source."
    )
    assert evidence["explicit_policy"], (
        "Cached-state detachment exists but is not exposed/documented as an intentional policy."
    )

    model = build_model(seed=2026, budget=4, force_detach=True)
    if supports_detach_override(model):
        model.train()
        encourage_writes(model)
        x = compatible_input(model, seq_len=8)
        reset_memory(model, require=False)
        invoke(model, x, reset=True)
        tensors = memory_tensors(model)
        graph_tensors = [
            tensor
            for tensor in tensors
            if tensor.dtype.is_floating_point and (tensor.requires_grad or tensor.grad_fn is not None)
        ]
        assert not graph_tensors, (
            "Detach mode was requested, but cached memory tensors remain connected to autograd."
        )


def test_14_12_trainable_cached_states_remain_connected() -> None:
    evidence = cache_graph_policy_evidence()
    assert evidence["trainable_path"], (
        "No explicit trainable cached-state path was found. Expose a configuration such as "
        "detach_cached_states=False or trainable_memory=True."
    )

    model = build_model(seed=2026, budget=4, force_detach=False)
    if supports_detach_override(model):
        model.train()
        encourage_writes(model)
        x = compatible_input(model, seq_len=8)
        reset_memory(model, require=False)
        invoke(model, x, reset=True)
        tensors = memory_tensors(model)
        connected = [
            tensor
            for tensor in tensors
            if tensor.dtype.is_floating_point and (tensor.requires_grad or tensor.grad_fn is not None)
        ]
        assert connected, (
            "Trainable-cache mode was requested, but no cached tensor remains connected "
            "to the autograd graph."
        )


def test_14_13_memory_reset_between_unrelated_sequences() -> None:
    model = build_model(seed=2026, budget=4)
    model.eval()
    encourage_writes(model)
    x = compatible_input(model, seq_len=10)
    reset_memory(model)
    with torch.no_grad():
        invoke(model, x, reset=False)
    populated_size = memory_size(model)
    assert populated_size is not None, "Memory size could not be inspected before reset."
    assert populated_size > 0, (
        "The test sequence did not populate memory; reset behavior cannot be verified."
    )

    reset_memory(model)
    cleared_size = memory_size(model)
    assert cleared_size == 0, (
        f"Memory was not cleared between unrelated sequences; size after reset={cleared_size}."
    )

    fresh = build_model(seed=2026, budget=4)
    fresh.load_state_dict(copy.deepcopy(model.state_dict()))
    fresh.eval()
    reset_memory(fresh, require=False)

    unrelated = mutate_suffix(x, 0)
    with torch.no_grad():
        output_reset = invoke(model, unrelated, reset=True)
        output_fresh = invoke(fresh, unrelated.clone(), reset=True)
    assert output_equal(output_reset, output_fresh), (
        "An unrelated sequence after reset differs from a fresh model with identical weights; "
        "memory state leaked across sequences."
    )
PY

log "Checking Python syntax."
"$PYTHON_BIN" -m py_compile tests/section14_runtime.py tests/test_section14_required.py

check_imports() {
    "$PYTHON_BIN" - <<'PY'
missing = []
for module in ("pytest", "torch", "numpy"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit(" ".join(missing))
PY
}

if ! missing_modules="$(check_imports 2>&1)"; then
    if [[ "$INSTALL_DEPS" == "1" ]]; then
        log "Required modules missing: $missing_modules"
        log "Attempting installation from the project configuration."
        "$PYTHON_BIN" -m pip install -e ".[dev]" \
            || "$PYTHON_BIN" -m pip install -e . \
            || die "Dependency installation failed. Activate the established project environment and rerun."
    else
        die "Required modules missing: $missing_modules"
    fi
fi

check_imports >/dev/null || die "pytest, torch, and numpy are required."

JUNIT_FILE="$REPO_ROOT/reports/evidence/junit/section14_unit_tests_${TIMESTAMP}.xml"
LOG_FILE="$REPO_ROOT/reports/evidence/logs/section14_unit_tests_${TIMESTAMP}.log"
RESULTS_FILE="$REPO_ROOT/reports/tables/section14_unit_test_results.csv"
REPORT_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_report.txt"
MANIFEST_FILE="$REPO_ROOT/reports/evidence/section14_unit_tests_manifest.json"

log "Discovering Section 14-specific test nodes."
mapfile -t SECTION14_TEST_TARGETS < <(
    "$PYTHON_BIN" - <<'PYTEST_TARGETS'
from __future__ import annotations

import ast
import re
from pathlib import Path

root = Path.cwd()
generated = root / "tests" / "test_section14_required.py"
targets = [str(generated.relative_to(root))]

pattern = re.compile(
    r"("
    r"strict_budget|memory.*budget|budget.*violat|"
    r"future_tokens|causal|"
    r"determin|same_seed|training_order|"
    r"synthetic.*seed|seed.*overlap|"
    r"hdfs.*block|block.*overlap|"
    r"imdb.*test|official.*test|split.*leak|"
    r"gradient|backpropagat.*controller|controller.*gradient|"
    r"graph_policy|cached_state|trainable_cache|detached_cache|"
    r"memory.*reset|reset.*memory"
    r")",
    re.IGNORECASE,
)

for path in sorted((root / "tests").rglob("test*.py")):
    if path.resolve() == generated.resolve():
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    relative = str(path.relative_to(root))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and pattern.search(node.name):
                targets.append(f"{relative}::{node.name}")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_") and pattern.search(child.name):
                        targets.append(f"{relative}::{node.name}::{child.name}")

for target in dict.fromkeys(targets):
    print(target)
PYTEST_TARGETS
)

if [[ "${#SECTION14_TEST_TARGETS[@]}" -eq 0 ]]; then
    die "No Section 14 tests were discovered."
fi

log "Section 14 pytest targets: ${#SECTION14_TEST_TARGETS[@]}"
printf '  %s\n' "${SECTION14_TEST_TARGETS[@]}"

log "Running the Section 14 pre-training gate."
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest \
    -q \
    -o addopts='' \
    "${SECTION14_TEST_TARGETS[@]}" \
    --junitxml="$JUNIT_FILE" \
    2>&1 | tee "$LOG_FILE"
PYTEST_EXIT_CODE="${PIPESTATUS[0]}"
set -e

export SECTION14_PYTEST_EXIT_CODE="$PYTEST_EXIT_CODE"
export SECTION14_JUNIT_FILE="$JUNIT_FILE"
export SECTION14_RESULTS_FILE="$RESULTS_FILE"
export SECTION14_REPORT_FILE="$REPORT_FILE"
export SECTION14_MANIFEST_FILE="$MANIFEST_FILE"
export SECTION14_LOG_FILE="$LOG_FILE"
export SECTION14_TIMESTAMP="$TIMESTAMP"
export SECTION14_SCRIPT_NAME="$SCRIPT_NAME"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

junit_file = Path(os.environ["SECTION14_JUNIT_FILE"])
results_file = Path(os.environ["SECTION14_RESULTS_FILE"])
report_file = Path(os.environ["SECTION14_REPORT_FILE"])
manifest_file = Path(os.environ["SECTION14_MANIFEST_FILE"])
log_file = Path(os.environ["SECTION14_LOG_FILE"])
pytest_exit_code = int(os.environ["SECTION14_PYTEST_EXIT_CODE"])
timestamp = os.environ["SECTION14_TIMESTAMP"]

categories = {
    "Budget correctness": ["test_14_01_"],
    "Causality": ["test_14_02_"],
    "Determinism": ["test_14_03_", "test_14_04_", "test_14_05_", "test_14_06_"],
    "Synthetic seed isolation": ["test_14_07_"],
    "HDFS block isolation": ["test_14_08_"],
    "IMDb official-test isolation": ["test_14_09_"],
    "Gradient flow": ["test_14_10_"],
    "Cached-state graph policy": ["test_14_11_", "test_14_12_"],
    "Memory reset": ["test_14_13_"],
}

testcases: list[dict[str, str]] = []
if junit_file.exists():
    root = ET.parse(junit_file).getroot()
    for case in root.iter("testcase"):
        name = case.attrib.get("name", "")
        classname = case.attrib.get("classname", "")
        time_value = case.attrib.get("time", "0")
        status = "PASS"
        detail = ""
        for child_status in ("failure", "error", "skipped"):
            child = case.find(child_status)
            if child is not None:
                status = child_status.upper()
                detail = (child.attrib.get("message") or child.text or "").strip()
                break
        testcases.append(
            {
                "classname": classname,
                "test_name": name,
                "status": status,
                "seconds": time_value,
                "detail": detail.replace("\n", " ")[:2000],
            }
        )

results_file.parent.mkdir(parents=True, exist_ok=True)
with results_file.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("classname", "test_name", "status", "seconds", "detail"),
    )
    writer.writeheader()
    writer.writerows(testcases)

category_status: dict[str, str] = {}
for category, prefixes in categories.items():
    matched = [
        case
        for case in testcases
        if any(case["test_name"].startswith(prefix) for prefix in prefixes)
    ]
    expected_count = len(prefixes)
    if len(matched) < expected_count:
        category_status[category] = "FAIL"
    elif all(case["status"] == "PASS" for case in matched):
        category_status[category] = "PASS"
    else:
        category_status[category] = "FAIL"

full_suite_status = "PASS" if pytest_exit_code == 0 else "FAIL"
all_required_pass = all(value == "PASS" for value in category_status.values())
final_decision = "GO" if all_required_pass and pytest_exit_code == 0 else "NO-GO"
section_status = "COMPLETE" if final_decision == "GO" else "INCOMPLETE"

failed = [
    case
    for case in testcases
    if case["status"] != "PASS"
]
lines = [
    "Section 14 — Unit Tests Required Before Training",
    f"Generated UTC: {timestamp}",
    "",
]
for category, status in category_status.items():
    lines.append(f"{category}: {status}")
lines.extend(
    [
        f"Complete pytest suite: {full_suite_status}",
        f"Pytest exit code: {pytest_exit_code}",
        "",
        f"Final decision: {final_decision}",
        f"Section 14: {section_status}",
        "",
        f"JUnit evidence: {junit_file.relative_to(report_file.parents[2])}",
        f"Detailed log: {log_file.relative_to(report_file.parents[2])}",
        f"Result table: {results_file.relative_to(report_file.parents[2])}",
    ]
)
if failed:
    lines.extend(["", "Failed or unresolved checks:"])
    for case in failed[:25]:
        detail = case["detail"] or "No failure detail was recorded."
        lines.append(f"- {case['test_name']}: {case['status']} — {detail}")

report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "section": "14",
    "generated_utc": timestamp,
    "pytest_exit_code": pytest_exit_code,
    "final_decision": final_decision,
    "section_status": section_status,
    "category_status": category_status,
    "python": sys.version,
    "platform": platform.platform(),
    "evidence": {
        "junit": str(junit_file),
        "log": str(log_file),
        "results_csv": str(results_file),
        "report": str(report_file),
    },
}
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print()
print(report_file.read_text(encoding="utf-8"))
PY

log "Evidence report: reports/evidence/section14_unit_tests_report.txt"
log "Result table: reports/tables/section14_unit_test_results.csv"
log "Detailed log: ${LOG_FILE#$REPO_ROOT/}"

if [[ "$PYTEST_EXIT_CODE" -ne 0 ]]; then
    printf '\nSECTION 14 RESULT: NO-GO\n'
    printf 'Open reports/evidence/section14_unit_tests_report.txt for the exact failed gate.\n'
    exit "$PYTEST_EXIT_CODE"
fi

printf '\nSECTION 14 RESULT: GO\n'
printf 'All required unit-test gates passed. Training may begin.\n'
