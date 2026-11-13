import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("v17_matrix", Path(__file__).with_name("run_v17_matrix.py"))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def test_matrix_only_contains_four_fixed_train_and_validation_routes(self):
        commands = matrix.stage_commands(Path("/candidate/runs/session"), Path("/weights"), Path("/inventory.json"), "a" * 64)
        self.assertEqual(len(commands), 13)
        self.assertEqual(len({name for name, _ in commands}), 13)
        trains = [args for name, args in commands if name.endswith("_train")]
        self.assertEqual(len(trains), 4)
        for command in trains:
            self.assertEqual(command[command.index("--mode") + 1], "validation")
        self.assertFalse(any("test_manifest" in str(args) or "formal.py" in str(args) for _, args in commands))
        self.assertEqual([name for name, _ in commands if name.endswith("_depth")], ["ours_dual_100_depth"])


if __name__ == "__main__":
    unittest.main()
