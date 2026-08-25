import os
import unittest
from pathlib import Path

from jhtdb_regimes.config import load_config
from jhtdb_regimes.jhtdb_client import JHTDBClient


@unittest.skipUnless(os.environ.get("JHTDB_ONLINE") == "1", "set JHTDB_ONLINE=1 to use the testing API")
class OnlineSmokeTest(unittest.TestCase):
    def test_eight_points_two_times(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "task0.yaml")
        result = JHTDBClient(cfg).smoke()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["points"], 8)
        self.assertEqual(len(result["times"]), 2)


if __name__ == "__main__":
    unittest.main()

