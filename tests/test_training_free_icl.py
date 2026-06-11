import json
import unittest

from super_auto_rubric.webshop.rubric_pool import RubricEntry
from super_auto_rubric.webshop.training_free_icl import (
    CriticRubricJudge,
    InContextRubricPolicy,
    is_action_tool_valid,
    run_training_free_icl_episode,
)
from super_auto_rubric.webshop.client import SyntheticWebShopClient


class FakeCompletion:
    def __init__(self, content):
        self.content = content
        self.raw = {}


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return FakeCompletion(self.responses.pop(0))


def rubric_entry(rubric_id="rubric_attr"):
    return RubricEntry(
        rubric_id=rubric_id,
        title="Attribute Neglect",
        natural_language_rule="Reward final purchases that explicitly satisfy every required user constraint.",
        polarity="positive",
        weight=1.0,
        severity=0.75,
        support_count=1,
        source_weakness_ids=["weak_1"],
        evidence_trajectory_ids=["traj_1"],
        cluster_key="attribute_neglect:positive",
        source_cluster_key="attribute_neglect",
        last_triggered_batch=0,
    )


class TrainingFreeICLTest(unittest.TestCase):
    def test_action_validity_checks_search_and_click_tools(self):
        self.assertTrue(
            is_action_tool_valid(
                "search[green mug]",
                {"has_search_bar": True, "clickables": ["search"]},
            )
        )
        self.assertTrue(
            is_action_tool_valid(
                "click[green jumpsuit]",
                {"has_search_bar": False, "clickables": ["green jumpsuit"]},
            )
        )
        self.assertFalse(
            is_action_tool_valid(
                "click[missing item]",
                {"has_search_bar": False, "clickables": ["green jumpsuit"]},
            )
        )

    def test_invalid_actor_response_falls_back_to_valid_scripted_action(self):
        rubric = rubric_entry()
        policy = InContextRubricPolicy(
            chat_client=FakeChatClient(["not json"]),
            actor_model="actor",
            rubrics=[rubric],
        )
        decision = policy.choose_decision(
            "WebShop [SEP] Search",
            {"has_search_bar": True, "clickables": ["search"]},
            "Find a green mug.",
        )
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.action, "search[green mug.]")
        self.assertFalse(decision.parse_ok)
        self.assertTrue(decision.tool_valid)

    def test_training_free_reward_combines_task_validity_and_critic_scores(self):
        rubric = rubric_entry()
        actor_responses = [
            json.dumps({"action": "search[green jumpsuit]", "rationale": "search", "rubric_focus": ["rubric_attr"]}),
            json.dumps({"action": "click[green jumpsuit]", "rationale": "inspect", "rubric_focus": ["rubric_attr"]}),
            json.dumps({"action": "click[small]", "rationale": "select", "rubric_focus": ["rubric_attr"]}),
        ]
        critic_response = json.dumps(
            {
                "scores": [
                    {
                        "rubric_id": "rubric_attr",
                        "score": -0.25,
                        "explanation": "missed the requested size",
                    }
                ]
            }
        )
        chat_client = FakeChatClient(actor_responses + [critic_response])
        policy = InContextRubricPolicy(
            chat_client=chat_client,
            actor_model="actor",
            rubrics=[rubric],
        )
        judge = CriticRubricJudge(
            chat_client=chat_client,
            critic_model="critic",
            rubrics=[rubric],
        )
        episode, breakdown = run_training_free_icl_episode(
            SyntheticWebShopClient(),
            policy,
            judge,
            split="test",
            max_steps=3,
            session_id=0,
            seed=1,
            actor_model="actor",
            critic_model="critic",
            rubric_version="rubric-test",
        )

        self.assertEqual(episode.trajectory.step_count, 3)
        self.assertEqual(breakdown.format_tool_validity_reward, 1.0)
        self.assertEqual(breakdown.critic_rubric_judged_score_sum, -0.25)
        self.assertEqual(
            breakdown.combined_reward,
            episode.trajectory.final_reward + 1.0 - 0.25,
        )


if __name__ == "__main__":
    unittest.main()
