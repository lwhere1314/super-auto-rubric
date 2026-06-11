import unittest

from super_auto_rubric.webshop.trajectory import Trajectory, TrajectoryStep


class TrajectorySchemaTest(unittest.TestCase):
    def test_trajectory_schema_keeps_required_metadata(self) -> None:
        trajectory = Trajectory.start(
            instruction_text="Find a green jumpsuit.",
            split="smoke",
            model="scripted",
            policy="test-policy",
            prompt_version="prompt-v0",
            rubric_version="rubric-v0",
            seed=7,
            session_id=3,
            metadata={"env_idx": 99},
        )
        trajectory.add_step(
            TrajectoryStep(
                step_index=0,
                observation_before="Search",
                available_actions={"has_search_bar": True, "clickables": ["search"]},
                action="search[green jumpsuit]",
                reward=0.0,
                done=False,
                observation_after="Results",
                state={
                    "url": "synthetic://0",
                    "html": "Search",
                    "instruction_text": "Find a green jumpsuit.",
                },
                info={"score": 0.0},
            )
        )

        data = trajectory.to_dict()
        reloaded = Trajectory.from_dict(data)

        self.assertEqual(reloaded.schema_version, "webshop-trajectory-v1")
        self.assertEqual(reloaded.prompt_version, "prompt-v0")
        self.assertEqual(reloaded.rubric_version, "rubric-v0")
        self.assertEqual(reloaded.metadata["env_idx"], 99)
        self.assertTrue(reloaded.steps[0].available_actions["has_search_bar"])


if __name__ == "__main__":
    unittest.main()
