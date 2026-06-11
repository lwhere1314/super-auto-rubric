import tempfile
import unittest
from pathlib import Path

from super_auto_rubric.webshop.baseline import (
    run_episode,
    save_batch_results,
    summarize_episode_results,
)
from super_auto_rubric.webshop.client import SyntheticWebShopClient
from super_auto_rubric.webshop.policies import ScriptedWebShopPolicy
from super_auto_rubric.webshop.rubric_pool import RubricPool
from super_auto_rubric.webshop.trajectory import load_trajectories
from super_auto_rubric.webshop.weakness import mine_batch


class WebShopLocalLoopTest(unittest.TestCase):
    def test_synthetic_episode_persists_reloadable_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_episode(
                SyntheticWebShopClient(),
                ScriptedWebShopPolicy(),
                split="smoke",
                max_steps=8,
                session_id=0,
                seed=42,
            )

            metrics = save_batch_results(tmp_path, [result])
            loaded = load_trajectories(tmp_path)

            self.assertEqual(metrics["episodes"], 1)
            self.assertEqual(metrics["average_steps"], 3)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0].instruction_text.startswith("Find a medium green"))
            self.assertTrue(loaded[0].steps[-1].done)

    def test_weakness_mining_and_rubric_pool_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_episode(
                SyntheticWebShopClient(),
                ScriptedWebShopPolicy(),
                split="baseline",
                max_steps=8,
                session_id=0,
                seed=42,
            )

            weaknesses = mine_batch([result.trajectory])
            self.assertTrue(weaknesses)
            self.assertIn("attribute_neglect", {item.label for item in weaknesses})

            pool = RubricPool(max_active_rubrics=5)
            pool.upsert_from_weaknesses(weaknesses, batch_id=0)

            self.assertEqual(len(pool.active), 2)
            entries = list(pool.active.values())
            self.assertEqual({entry.polarity for entry in entries}, {"negative", "positive"})
            negative_entry = next(entry for entry in entries if entry.polarity == "negative")
            positive_entry = next(entry for entry in entries if entry.polarity == "positive")
            self.assertLess(negative_entry.weight, 0)
            self.assertGreater(positive_entry.weight, 0)
            self.assertIn("required user constraints", negative_entry.natural_language_rule)

            active_path = tmp_path / "active.jsonl"
            retired_path = tmp_path / "retired.jsonl"
            pool.save(active_path, retired_path)
            loaded = RubricPool.load(active_path)
            self.assertEqual(len(loaded.active), 2)

    def test_batch_metrics_include_invalid_and_loop_rates(self) -> None:
        results = [
            run_episode(
                SyntheticWebShopClient(),
                ScriptedWebShopPolicy(),
                split="baseline",
                max_steps=8,
                session_id=idx,
                seed=idx,
            )
            for idx in range(2)
        ]

        metrics = summarize_episode_results(results)

        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["invalid_action_rate"], 0)
        self.assertIn("loop_rate", metrics)


if __name__ == "__main__":
    unittest.main()
