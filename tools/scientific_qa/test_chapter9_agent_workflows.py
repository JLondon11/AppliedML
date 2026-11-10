"""Verify observed deterministic tool-agent workflows on held-out real images."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"chapters/generative_ai_part_ii/code/figures06_07_agent_workflows"))
from experiment import evaluate

class ToolWorkflowTests(unittest.TestCase):
    def test_recorded_events_are_consistent(self):
        rng=np.random.default_rng(23)
        p={name:rng.dirichlet(np.ones(10),size=12) for name in ("linear","forest","neighbors")}
        outcomes,events=evaluate(p,.6)
        self.assertEqual(set(outcomes),{"planner-executor","supervisor-worker","peer-vote"})
        self.assertTrue(all(len(v["prediction"])==12 for v in outcomes.values()))
        for event in events:
            self.assertTrue(0<=event["prediction"]<=9)
            self.assertTrue(0<=event["confidence"]<=1)
        self.assertEqual(sum(x["event"]=="final_answer" for x in events),36)

    def test_fallback_only_below_threshold(self):
        p={name:np.full((4,10),.01) for name in ("linear","forest","neighbors")}
        p["linear"][:,0]=.91
        p["forest"][:,1]=.91
        p["neighbors"][:,2]=.91
        results,events=evaluate(p,.8)
        self.assertFalse(results["planner-executor"]["consulted"].any())
        self.assertFalse(results["supervisor-worker"]["consulted"].any())
        self.assertEqual(sum(x["event"]=="fallback_tool" for x in events),0)

if __name__=="__main__":
    unittest.main()
