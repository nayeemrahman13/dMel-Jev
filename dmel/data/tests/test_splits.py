"""Split-assignment tests: cell-based holdouts, seed ranges, eval-only slots."""

from dmel.data.scenarios import (
    VOICE_PAIRS,
    _eval_only_subset,
    _slot_class,
    bank_pools,
    build_pools,
    sample_seed,
    split_for_cell,
)

SALT = "unit|splits"


def make_pools(scale: str = "full"):
    return build_pools(salt=SALT, scale=scale)


class TestEvalOnlySubsets:
    def test_voice_pairs_held_out_about_30pct(self):
        pools = make_pools()
        frac = len(pools.eval_pairs) / len(VOICE_PAIRS)
        assert 0.15 <= frac <= 0.5, frac
        assert pools.eval_pairs  # at least one held-out pair

    def test_template_slots_held_out_within_each_class(self):
        pools = make_pools()
        for cls, pool in bank_pools("full").items():
            held = pools.eval_slots[cls]
            if len(pool) >= 3:
                assert held, f"class {cls} lost its eval-only slots"
                frac = len(held) / len(pool)
                assert 0.15 <= frac <= 0.5, (cls, frac)
            else:
                assert not held  # tiny pools stay usable in every split

    def test_eval_only_subset_deterministic(self):
        a = _eval_only_subset(10, "k")
        b = _eval_only_subset(10, "k")
        assert a == b
        assert len(a) == 3  # round(10 * 0.3)


class TestCellSplit:
    def test_heldout_components_land_in_eval(self):
        pools = make_pools()
        for vp in pools.eval_pairs:
            for scenario in ("interruption", "normal_turn"):
                assert split_for_cell(scenario, vp, 0, pools) in ("val", "test")

    def test_train_cells_use_70_15_15_hash(self):
        pools = make_pools()
        counts = {"train": 0, "val": 0, "test": 0}
        n_slots = 20
        for scenario in ("interruption", "hesitation"):
            for vp in range(len(VOICE_PAIRS)):
                for slot in range(n_slots):
                    if vp in pools.eval_pairs or slot in pools.eval_slots[_slot_class(scenario)]:
                        continue  # forced-eval cells excluded from the 70/15/15 hash
                    counts[split_for_cell(scenario, vp, slot, pools)] += 1
        total = sum(counts.values())
        assert counts["train"] / total >= 0.6
        assert counts["val"] / total >= 0.05
        assert counts["test"] / total >= 0.05

    def test_split_is_total(self):
        pools = make_pools()
        for scenario in ("interruption", "backchannel", "noise"):
            for vp in range(len(VOICE_PAIRS)):
                for slot in range(10):
                    assert split_for_cell(scenario, vp, slot, pools) in ("train", "val", "test")


class TestSeedRanges:
    def test_train_below_million_eval_above(self):
        for index in (0, 1, 999):
            assert 0 <= sample_seed(42, index, "train") < 1_000_000
            assert 1_000_000 <= sample_seed(42, index, "val") < 2_000_000
            assert 1_000_000 <= sample_seed(42, index, "test") < 2_000_000
