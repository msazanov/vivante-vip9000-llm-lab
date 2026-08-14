"""Phase-level promotion tests using only committed E055 raw bundles."""

from __future__ import annotations

import unittest

from tests.test_e055_raw_bundle import BundleFixture
from tooling.e055_q1_hotcold import (
    CPUS,
    MODES,
    PMU_GROUP_CONFIGS,
    WORKING_SET_BYTES,
    infer_bottleneck,
)


def complete_matrix_specs(
    *, default_cold_ns: int = 1_050_000, include_adversarial_cell: bool = True,
) -> list[dict]:
    """Build the exact 1,260-run matrix with one adversarial paired cell."""

    specs: list[dict] = []
    adversarial_hot = [1_000_000, 1_000_000, 1_000_000, 100_000_000, 100_000_000]
    adversarial_cold = [100_000_000, 100_000_000, 1_000_000, 100_000_000, 100_000_000]
    for target in WORKING_SET_BYTES:
        for cpu in CPUS:
            for mode in MODES:
                for pmu_group in PMU_GROUP_CONFIGS:
                    cell_name = f"{target}-{cpu}-{mode}-{pmu_group}"
                    for pair_index in range(1, 6):
                        pair_id = f"pair-{cell_name}-{pair_index}"
                        pair_order = (
                            "hot_then_cold" if pair_index % 2 else "cold_then_hot"
                        )
                        adversarial = include_adversarial_cell and (
                            target == WORKING_SET_BYTES[0]
                            and cpu == CPUS[0]
                            and mode == "full_dotprod"
                            and pmu_group == "core"
                        )
                        hot_ns = (
                            adversarial_hot[pair_index - 1]
                            if adversarial else 1_000_000
                        )
                        cold_ns = (
                            adversarial_cold[pair_index - 1]
                            if adversarial else default_cold_ns
                        )
                        for state, ns_per_call in (
                            ("hot_repeat", hot_ns),
                            ("cold_conditioned", cold_ns),
                        ):
                            order_index = 1 if (
                                (pair_order == "hot_then_cold" and state == "hot_repeat")
                                or (
                                    pair_order == "cold_then_hot"
                                    and state == "cold_conditioned"
                                )
                            ) else 2
                            specs.append({
                                "run_id": f"run-{cell_name}-{pair_index}-{state}",
                                "pair_id": pair_id,
                                "pair_index": pair_index,
                                "pair_order": pair_order,
                                "order_index": order_index,
                                "build_name": "O3",
                                "mode": mode,
                                "cache_state": state,
                                "cpu": cpu,
                                "target_working_set_bytes": target,
                                "pmu_group": pmu_group,
                                "ns_per_call": ns_per_call,
                            })
    return specs


class E055SealedPromotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = BundleFixture(complete_matrix_specs())
        cls.manifest_size = cls.fixture.manifest.stat().st_size
        cls.result = infer_bottleneck(cls.fixture.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def test_complete_matrix_fits_manifest_cap_and_promotes_only_sealed_rows(self) -> None:
        self.assertLess(self.manifest_size, 16 * 1024 * 1024)
        self.assertEqual(self.result["matrix_qualified_rows"], 1260)
        self.assertTrue(self.result["matrix_complete"])
        self.assertTrue(self.result["promotion"]["eligible"])

    def test_median_is_computed_from_pairs_not_independent_state_medians(self) -> None:
        record = next(
            item for item in self.result["records"]
            if item["cpu"] == 0
            and item["target_working_set_bytes"] == WORKING_SET_BYTES[0]
            and item["pmu_group"] == "core"
        )
        self.assertEqual(record["full_cold_over_hot"], 1.0)
        self.assertEqual(
            [item["cold_over_hot"] for item in record["pair_penalties"]["full_dotprod"]],
            [100.0, 100.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(
            record["pmu_event_median_cold_over_hot"]["full_dotprod"]["cpu_cycles"],
            250.0,
        )

    def test_incomplete_committed_bundle_cannot_reach_promotion(self) -> None:
        with BundleFixture() as fixture:
            with self.assertRaisesRegex(ValueError, "complete documented"):
                infer_bottleneck(fixture.manifest)

    def test_four_percent_projected_gain_threshold_is_enforced_on_sealed_phase(self) -> None:
        specs = complete_matrix_specs(
            default_cold_ns=1_040_000, include_adversarial_cell=False,
        )
        with BundleFixture(specs) as fixture:
            result = infer_bottleneck(fixture.manifest)
        self.assertEqual(result["promotion"]["threshold_fraction"], 0.04)
        self.assertAlmostEqual(
            result["promotion"]["projected_q1_gain_fraction"],
            1.0 - 1.0 / 1.04,
        )
        self.assertFalse(result["promotion"]["eligible"])


if __name__ == "__main__":
    unittest.main()
