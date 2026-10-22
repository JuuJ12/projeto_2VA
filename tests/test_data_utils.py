import os
import tempfile
import unittest

import pandas as pd

from src.data_utils import resolve_test_dataset_paths


class ResolveTestDatasetPathsTests(unittest.TestCase):
    def test_resolves_windows_paths_on_posix_deployments(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            image_path = os.path.join(dataset_dir, "Apex Legends", "image_50.png")
            os.makedirs(os.path.dirname(image_path))
            open(image_path, "wb").close()

            test_df = pd.DataFrame(
                [
                    {
                        "path": r"C:\VScode\projeto_2VA\Apex Legends\image_50.png",
                        "label": "Apex Legends",
                    }
                ]
            )

            paths = resolve_test_dataset_paths(test_df, dataset_dir)

            self.assertEqual(paths, [image_path])


if __name__ == "__main__":
    unittest.main()
