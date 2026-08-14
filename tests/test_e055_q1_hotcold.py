"""Adversarial host-side contract tests for E055 qualification."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import unittest
from unittest import mock

import tooling.e055_q1_hotcold as e055
from tooling.e055_q1_hotcold import (
    CACHE_STATES,
    CPUS,
    MODES,
    WORKING_SET_BYTES,
    actual_working_set,
    cache_penalty_ratio,
    infer_bottleneck,
    logical_bytes_per_call,
    ns_per_traversal,
    paired_speed_delta,
    summarize_samples,
    traversals_per_second,
    validate_sample,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "38c66ad0241da4f9fcce541cda8edc219086cec5"
UPSTREAM_REF = (
    "https://github.com/ggml-org/llama.cpp/commit/"
    "38c66ad0241da4f9fcce541cda8edc219086cec5"
)
PMU_CONFIGS = {
    "core": {
        "cpu_cycles": "0x11",
        "instructions": "0x8",
        "stall_backend": "0x24",
    },
    "cache": {
        "l1d_cache_refill": "0x3",
        "l2d_cache_refill": "0x17",
        "l3d_cache_refill": "0x2a",
    },
    "memory": {"mem_access": "0x13", "bus_access": "0x19"},
}


def published_provenance(build_name: str = "O3") -> dict:
    disassembly = json.loads(
        (ROOT / "experiments/E055-q1-hot-cold/data/disassembly-review.json")
        .read_text(encoding="utf-8")
    )
    build = next(item for item in disassembly["builds"] if item["name"] == build_name)
    return {
        "build_name": build_name,
        "source_sha256": disassembly["provenance"]["source_sha256"],
        "binary_sha256": build["binary_sha256"],
        "compiler_sha256": disassembly["provenance"]["compiler_sha256"],
        "compiler_id": disassembly["provenance"]["compiler_id"],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_ref": UPSTREAM_REF,
        "upstream_repack_sha256": (
            "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d"
        ),
    }


def bind_harness_result(sample: dict) -> None:
    result = {
        "schema": "e055-q1-hot-cold-harness/v1",
        "mode": sample["mode"],
        "cache_state": sample["cache_state"],
        "cpu": sample["cpu"],
        "target_working_set_bytes": sample["target_working_set_bytes"],
        "actual_working_set_bytes": sample["actual_working_set_bytes"],
        "blocks": sample["blocks"],
        "iterations": sample["iterations"],
        "calls": sample["calls"],
        "elapsed_ns": sample["elapsed_ns"],
        "checksum": sample["checksum"],
        "golden_pass": sample["golden_pass"],
        "golden_cases": sample["golden_cases"],
        "conditioning": copy.deepcopy(sample["cold_conditioning"]),
        "provenance": copy.deepcopy(sample["provenance"]),
    }
    sample["harness_result"] = result
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    sample["harness_result_sha256"] = hashlib.sha256(canonical).hexdigest()


def qualified_sample(
    cache_state: str = "hot_repeat",
    calls: int = 250,
    *,
    mode: str = "full_dotprod",
    pair_index: int = 1,
    cpu: int = 6,
    event_group: str = "core",
    target_bytes: int = 64 * 1024,
) -> dict:
    layout = actual_working_set(target_bytes)
    pair_order = "hot_then_cold" if pair_index % 2 else "cold_then_hot"
    order_index = 1 if (
        (pair_order == "hot_then_cold" and cache_state == "hot_repeat") or
        (pair_order == "cold_then_hot" and cache_state == "cold_conditioned")
    ) else 2
    configs = PMU_CONFIGS[event_group]
    run_prefix = f"c{cpu}-s{target_bytes}-{mode}-{event_group}-p{pair_index}"
    if cache_state == "cold_conditioned":
        requested = 64 * 1024 * 1024
        conditioning = {
            "strategy": "verified_write_read_each_64B_line",
            "requested_bytes": requested,
            "actual_bytes": requested,
            "line_bytes": 64,
            "lines_touched": requested // 64,
            "checksum": "0xe055a733",
            "verified_touched": True,
        }
    else:
        requested = layout["actual_bytes"]
        conditioning = {
            "strategy": "verified_kernel_warmup",
            "requested_bytes": requested,
            "actual_bytes": requested,
            "line_bytes": 64,
            "lines_touched": (requested + 63) // 64,
            "checksum": "0xe055a734",
            "verified_touched": True,
            "warmup_calls": 16,
        }
    checksum_seed = hashlib.sha256(
        f"{run_prefix}-{cache_state}-{calls}".encode()
    ).hexdigest()[:16]
    sample = {
        "schema": "e055-q1-hot-cold/v2",
        "run_id": f"run-{run_prefix}-{cache_state}",
        "pair_id": f"pair-{run_prefix}",
        "pair_index": pair_index,
        "pair_order": pair_order,
        "order_index": order_index,
        "mode": mode,
        "cache_state": cache_state,
        "cpu": cpu,
        "pinned_cpu": cpu,
        "affinity_cpus": [cpu],
        "cpu_start": cpu,
        "cpu_end": cpu,
        "cpu_migration_count": 0,
        "golden_pass": True,
        "golden_cases": 18,
        "cold_conditioning": conditioning,
        "target_working_set_bytes": target_bytes,
        "actual_working_set_bytes": layout["actual_bytes"],
        "blocks": layout["blocks"],
        "iterations": calls,
        "elapsed_ns": calls * 1_000_000,
        "calls": calls,
        "checksum": f"0x{checksum_seed}",
        "sync": {"requested": True, "started": True, "acknowledged": True,
                 "ended": True, "sequence": "S/A/E"},
        "pmu": {
            "schema_version": "e049c-arm-pmu/v2",
            "status": "ok",
            "sample_valid": True,
            "event_source": "armv8_pmuv3_raw_config",
            "event_group": event_group,
            "event_group_size": len(configs),
            "values_are_event_counts": True,
            "events": [
                {"name": name, "config": config, "support": "supported",
                 "sample_valid": True, "value": 10, "running_ratio": 1.0}
                for name, config in configs.items()
            ],
        },
        "thermal": {"readable": True, "tripped": False, "max_temp_c": 61.0,
                    "limit_c": 80.0},
        "provenance": published_provenance(),
        "exit_code": 0,
    }
    bind_harness_result(sample)
    return sample


def qualified_matrix(penalties: dict[str, list[float]] | None = None) -> list[dict]:
    penalties = penalties or {mode: [2.0] * 5 for mode in MODES}
    rows = []
    for target_bytes in WORKING_SET_BYTES:
        for cpu in CPUS:
            for event_group in PMU_CONFIGS:
                for mode in MODES:
                    for pair_index in range(1, 6):
                        hot = qualified_sample(
                            "hot_repeat", 250, mode=mode, pair_index=pair_index,
                            cpu=cpu, event_group=event_group, target_bytes=target_bytes,
                        )
                        cold = qualified_sample(
                            "cold_conditioned", 1, mode=mode, pair_index=pair_index,
                            cpu=cpu, event_group=event_group, target_bytes=target_bytes,
                        )
                        hot["elapsed_ns"] = 250 * 1_000_000
                        cold["elapsed_ns"] = round(
                            penalties[mode][pair_index - 1] * 1_000_000
                        )
                        bind_harness_result(hot)
                        bind_harness_result(cold)
                        rows.extend((hot, cold))
    return rows


class E055Q1HotColdContractTest(unittest.TestCase):
    def test_working_set_and_declared_bytes_match_stock_carrier(self) -> None:
        result = actual_working_set(64 * 1024)
        self.assertEqual(result["actual_bytes"] % 208, 0)
        counts = logical_bytes_per_call(40)
        self.assertEqual(counts["total_input_bytes"], 40 * 208)
        self.assertEqual(counts["dot_products"], 40 * 512)

    def test_shape_helpers_reject_bool_float_and_non_integer(self) -> None:
        for value in (True, 1.0, "65536"):
            with self.subTest(function="actual_working_set", value=value):
                with self.assertRaises(ValueError):
                    actual_working_set(value)
            with self.subTest(function="logical_bytes_per_call", value=value):
                with self.assertRaises(ValueError):
                    logical_bytes_per_call(value)

    def test_working_set_rejects_uint64_rounding_overflow(self) -> None:
        with self.assertRaises(ValueError):
            actual_working_set((1 << 64) - 207)

    def test_working_set_rejects_shapes_the_cpp_harness_cannot_allocate(self) -> None:
        maximum_cpp_blocks = (2**31 - 1) // 128
        with self.assertRaisesRegex(ValueError, r"C\+\+|native blocks"):
            actual_working_set(maximum_cpp_blocks * 208 + 1)

    def test_public_matrix_contains_exact_binary_12_5_mib(self) -> None:
        self.assertEqual(WORKING_SET_BYTES[-1], 13_107_200)

    def test_upstream_repack_is_pinned_to_reviewed_hash(self) -> None:
        self.assertEqual(getattr(e055, "UPSTREAM_REPACK_SHA256", None),
                         published_provenance()["upstream_repack_sha256"])

    def test_pmu_configs_match_exact_e049c_source_definitions(self) -> None:
        source = (ROOT / "tooling/a733_pmu_exec.c").read_text(encoding="utf-8")
        for configs in PMU_CONFIGS.values():
            for name, config in configs.items():
                numeric = int(config, 16)
                self.assertRegex(
                    source,
                    re.compile(rf'\{{"{re.escape(name)}",\s*0x0*{numeric:x}\b', re.I),
                )

    def test_normalization_exactly_handles_250_hot_calls_vs_one_cold_call(self) -> None:
        hot = qualified_sample("hot_repeat", 250)
        cold = qualified_sample("cold_conditioned", 1)
        self.assertEqual(ns_per_traversal(hot), 1_000_000.0)
        self.assertEqual(ns_per_traversal(cold), 1_000_000.0)
        self.assertEqual(traversals_per_second(hot), 1000.0)
        self.assertEqual(cache_penalty_ratio(hot, cold), 1.0)

    def test_paired_delta_normalizes_calls_and_rejects_unlike_modes(self) -> None:
        reference = qualified_sample(calls=250)
        candidate = qualified_sample(calls=1)
        candidate["elapsed_ns"] = 900_000
        self.assertAlmostEqual(paired_speed_delta(reference, candidate), 0.1)
        candidate["mode"] = "unpack_scale"
        with self.assertRaises(ValueError):
            paired_speed_delta(reference, candidate)

    def test_fully_qualified_sample_is_bound_to_publication_artifacts(self) -> None:
        self.assertTrue(any(
            "sealed raw-bundle" in error for error in validate_sample(qualified_sample())
        ))
        self.assertTrue(any(
            "sealed raw-bundle" in error
            for error in validate_sample(qualified_sample("cold_conditioned", 1))
        ))
        with self.assertRaises(TypeError):
            validate_sample(qualified_sample(), {"source_sha256": "4" * 64})

    def test_strict_validator_rejects_rereviewed_forgery_cases(self) -> None:
        mutations = {
            "forged PMU config": lambda s: s["pmu"]["events"][0].update(config="0xff"),
            "all-zero source": lambda s: s["provenance"].update(source_sha256="0" * 64),
            "arbitrary binary": lambda s: s["provenance"].update(binary_sha256="4" * 64),
            "forged compiler": lambda s: s["provenance"].update(compiler_id="forged"),
            "Boolean pinned CPU": lambda s: s.update(pinned_cpu=True),
            "float affinity CPU": lambda s: s.update(affinity_cpus=[6.0]),
            "Boolean migration count": lambda s: s.update(cpu_migration_count=False),
            "Boolean order index": lambda s: s.update(order_index=True),
            "float PMU group size": lambda s: s["pmu"].update(event_group_size=3.0),
            "Boolean thermal maximum": lambda s: s["thermal"].update(max_temp_c=True),
            "conditioning coverage gap": lambda s: s["cold_conditioning"].update(
                lines_touched=s["cold_conditioning"]["lines_touched"] - 1),
            "conditioning forged checksum": lambda s: s["cold_conditioning"].update(
                checksum="0x0"),
            "wrong golden cases": lambda s: s.update(golden_cases=17),
            "forged output checksum": lambda s: s.update(checksum="0x1234"),
            "Boolean exit code": lambda s: s.update(exit_code=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                sample = qualified_sample("cold_conditioned", 1)
                mutate(sample)
                self.assertTrue(validate_sample(sample), name)

    def test_existing_strict_fail_closed_cases_remain_rejected(self) -> None:
        mutations = {
            "sync=false": lambda s: s["sync"].update(requested=False),
            "empty events": lambda s: s["pmu"].update(events=[]),
            "negative PMU count": lambda s: s["pmu"]["events"][0].update(value=-1),
            "Boolean PMU count": lambda s: s["pmu"]["events"][0].update(value=True),
            "integer ratio": lambda s: s["pmu"]["events"][0].update(running_ratio=1),
            "multiplexed PMU": lambda s: s["pmu"]["events"][0].update(running_ratio=.999),
            "thermal trip": lambda s: s["thermal"].update(tripped=True),
            "CPU migration": lambda s: s.update(cpu_end=0, cpu_migration_count=1),
            "uppercase checksum": lambda s: s.update(checksum="0xAB"),
            "unverified conditioning": lambda s: s["cold_conditioning"].update(
                verified_touched=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                sample = qualified_sample()
                mutate(sample)
                self.assertTrue(validate_sample(sample), name)

    def test_cold_qualification_rejects_more_than_one_traversal(self) -> None:
        sample = qualified_sample("cold_conditioned", 2)
        self.assertTrue(any("sealed raw-bundle" in error for error in validate_sample(sample)))

    def test_analyzer_rejects_complete_but_fabricated_mapping_matrix(self) -> None:
        adversarial = {
            mode: [1.0, 100.0, 100.0, 1.0, 0.01] for mode in MODES
        }
        with self.assertRaisesRegex(TypeError, "committed raw-bundle manifest"):
            infer_bottleneck(qualified_matrix(adversarial))

    def test_public_promotion_unconditionally_requires_global_publication_state(self) -> None:
        from tests.test_e055_raw_bundle import BundleFixture

        with BundleFixture() as fixture, mock.patch.object(
            e055, "_require_global_publication_state",
            side_effect=ValueError("global publication state rejected"),
        ) as gate, self.assertRaisesRegex(ValueError, "global publication state rejected"):
            infer_bottleneck(fixture.manifest)
        gate.assert_called_once_with()

    def test_analyzer_rejects_invalid_pairs_and_incomplete_matrix(self) -> None:
        cases = {}
        invalid = qualified_matrix()
        invalid[0]["sync"]["ended"] = False
        cases["invalid strict sample"] = invalid
        duplicate_run = qualified_matrix()
        duplicate_run[1]["run_id"] = duplicate_run[0]["run_id"]
        cases["duplicate run ID"] = duplicate_run
        missing_half = qualified_matrix()[:-1]
        cases["missing pair or matrix half"] = missing_half
        incomplete_cell = qualified_matrix()
        incomplete_cell = [row for row in incomplete_cell if not (
            row["target_working_set_bytes"] == WORKING_SET_BYTES[-1]
            and row["cpu"] == CPUS[-1] and row["mode"] == MODES[-1]
            and row["pmu"]["event_group"] == "memory"
        )]
        cases["incomplete documented matrix"] = incomplete_cell
        mixed_build = qualified_matrix()
        lto = published_provenance("O3-flto")
        for row in mixed_build:
            if row["target_working_set_bytes"] == WORKING_SET_BYTES[0] and \
                    row["cpu"] == CPUS[0] and row["mode"] == MODES[0] and \
                    row["pmu"]["event_group"] == "core":
                row["provenance"] = copy.deepcopy(lto)
                bind_harness_result(row)
        cases["mixed allowed builds in one phase"] = mixed_build
        for name, rows in cases.items():
            with self.subTest(name=name), self.assertRaises(TypeError):
                infer_bottleneck(rows)

    def test_analyzer_enforces_four_percent_promotion_threshold(self) -> None:
        for penalty in (1.04, 1.05):
            with self.subTest(penalty=penalty), self.assertRaises(TypeError):
                infer_bottleneck(
                    qualified_matrix({mode: [penalty] * 5 for mode in MODES})
                )

    def test_summary_and_public_matrix(self) -> None:
        summary = summarize_samples([100.0, 110.0, 90.0, 100.0, 100.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 100.0)
        self.assertEqual(MODES, ("packed_stream", "unpack_scale", "full_dotprod"))
        self.assertEqual(CACHE_STATES, ("hot_repeat", "cold_conditioned"))
        self.assertEqual(len(WORKING_SET_BYTES), 7)


if __name__ == "__main__":
    unittest.main()
