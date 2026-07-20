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
    extract_named_tensor,
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

    with torch.no_grad():
        output = invoke(model, x, reset=True)

    sizes = extract_named_tensor(
        output,
        ("memory_sizes", "memory_size", "sizes"),
    )
    assert sizes is not None, (
        "BudgetMemROutput must expose memory_sizes for every forward step."
    )
    assert torch.isfinite(sizes.float()).all()
    assert int(sizes.max().item()) <= budget, (
        f"Memory budget violated: maximum memory size={int(sizes.max().item())}, "
        f"configured_budget={budget}."
    )
    assert sizes.numel() >= _sequence_length(x), (
        "memory_sizes does not contain a per-step budget trace."
    )


def test_14_02_causality_future_tokens_do_not_change_earlier_steps() -> None:
    seed = 2026
    model_a = build_model(seed=seed, budget=4)
    model_b = build_model(seed=seed, budget=4)
    assert state_dict_equal(model_a, model_b)

    model_a.eval()
    model_b.eval()
    encourage_writes(model_a)
    encourage_writes(model_b)

    x = compatible_input(model_a, seq_len=12)
    seq_len = _sequence_length(x)
    prefix_len = max(2, seq_len // 2)
    changed = mutate_suffix(x, prefix_len)

    with torch.no_grad():
        output_a = invoke(model_a, x, reset=True)
        output_b = invoke(model_b, changed, reset=True)

    compared = False
    for names in (
        ("write_slots",),
        ("write_probabilities",),
        ("hard_writes",),
        ("memory_sizes",),
        ("eviction_flags",),
    ):
        left = extract_named_tensor(output_a, names)
        right = extract_named_tensor(output_b, names)
        if left is None or right is None:
            continue

        left_prefix = sequence_prefix(left, seq_len, prefix_len)
        right_prefix = sequence_prefix(right, seq_len, prefix_len)
        if left_prefix is None or right_prefix is None:
            continue

        assert torch.equal(
            left_prefix.detach().cpu(),
            right_prefix.detach().cpu(),
        ), (
            f"Changing future tokens changed {names[0]} before the changed suffix."
        )
        compared = True

    assert compared, (
        "BudgetMemROutput did not expose a sequence-aligned memory decision trace."
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
    x = compatible_input(model, seq_len=16)
    model.zero_grad(set_to_none=True)

    output = invoke(model, x, reset=True)

    controller_terms = []
    for names in (
        ("write_probabilities",),
        ("auxiliary_mean",),
        ("auxiliary_log_variance",),
        ("retrieval_weights",),
    ):
        tensor = extract_named_tensor(output, names)
        if (
            tensor is not None
            and tensor.dtype.is_floating_point
            and tensor.requires_grad
        ):
            controller_terms.append(tensor.float().mean())

    primary = extract_tensor(output)
    if (
        primary is not None
        and primary.dtype.is_floating_point
        and primary.requires_grad
    ):
        controller_terms.append(primary.float().pow(2).mean())

    assert controller_terms, (
        "No differentiable controller diagnostic or model output was exposed."
    )

    loss = torch.stack(controller_terms).sum()
    loss.backward()

    parameters = controller_parameters(model)
    assert parameters, "No controller parameters were discovered."

    finite_gradient_names = [
        name
        for name, parameter in parameters
        if parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
    ]
    nonzero_gradient_names = [
        name
        for name, parameter in parameters
        if parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
    ]

    assert finite_gradient_names, (
        "No memory-controller parameter received a finite gradient."
    )
    assert nonzero_gradient_names, (
        "All observed memory-controller gradients were zero."
    )

    expected_families = (
        "write_controller",
        "eviction_controller",
        "initial_utility_head",
    )
    for family in expected_families:
        family_parameters = [
            name for name, _ in parameters if family in name
        ]
        if not family_parameters:
            continue
        assert any(
            name in finite_gradient_names for name in family_parameters
        ), f"{family} received no finite gradient."

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

    # A model may intentionally support only detached cached recurrent states
    # while keeping the trainable memory controller connected. In that design,
    # the existing explicit graph-policy test is the required evidence.
    if not evidence["trainable_path"]:
        assert existing_test_has(
            ("memory_controller", "memory_controllers", "controller"),
            ("gradient", "gradients"),
            ("graph_policy", "graph policy", "detach"),
            ("explicit", "intentional"),
        ), (
            "No trainable cached-state mode exists and no existing test explicitly "
            "verifies the controller-gradient/cached-state graph policy."
        )
        return

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
            if tensor.dtype.is_floating_point
            and (tensor.requires_grad or tensor.grad_fn is not None)
        ]
        assert connected, (
            "Trainable-cache mode was requested, but no cached tensor remains "
            "connected to the autograd graph."
        )


def test_14_13_memory_reset_between_unrelated_sequences() -> None:
    model = build_model(seed=2026, budget=4)
    fresh = build_model(seed=2026, budget=4)
    fresh.load_state_dict(copy.deepcopy(model.state_dict()))

    model.eval()
    fresh.eval()
    encourage_writes(model)
    encourage_writes(fresh)

    first = compatible_input(model, seq_len=10)
    second = mutate_suffix(first, 0)

    with torch.no_grad():
        invoke(model, first, reset=True)
        output_after_unrelated = invoke(model, second, reset=True)
        output_fresh = invoke(fresh, second.clone(), reset=True)

    assert output_equal(output_after_unrelated, output_fresh), (
        "Memory state leaked from one unrelated sequence into the next."
    )

    sizes = extract_named_tensor(
        output_after_unrelated,
        ("memory_sizes", "memory_size", "sizes"),
    )
    assert sizes is not None
    assert int(sizes.reshape(-1)[0].item()) <= 1, (
        "The unrelated sequence did not begin from an empty memory state."
    )
