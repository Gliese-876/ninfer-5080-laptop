#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only CUDA runtime selection checks; all ldd calls are mocked.

Run: python -m unittest discover -s scripts -p test_launch_ninfer.py -v
No executable, real model, CUDA API, or GPU is started.
"""

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import launch_ninfer as launcher


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Path("/usr/local/cuda-13.4").resolve()
        self.binary = Path("/tmp/mock-ninfer-serve")
        self.environment = {"NINFER_CUDA": str(self.cuda),
                            "LD_LIBRARY_PATH": str(self.cuda) + "/lib64:/usr/lib/wsl/lib"}

    def evidence(self, output):
        result = subprocess.CompletedProcess(["ldd", str(self.binary)], 0, output, "")
        with patch.object(launcher.subprocess, "run", return_value=result) as run:
            evidence = launcher.runtime_evidence(self.binary, self.cuda, self.environment)
        run.assert_called_once_with(["ldd", str(self.binary)], capture_output=True,
                                    text=True, env=self.environment, check=True)
        return evidence

    def test_environment_removes_old_cuda_and_preserves_other_libraries(self):
        inherited = {
            "NINFER_CUDA": "/usr/local/cuda-13.3",
            "LD_LIBRARY_PATH": ":".join((
                "/usr/local/cuda-13.3/lib64", "/usr/lib/wsl/lib",
                "/usr/local/cuda-13.3/targets/x86_64-linux/lib",
                "/opt/custom-library/lib", "/usr/local/cuda/lib64/",
                "/usr/lib/wsl/lib", "",
            )),
            "UNCHANGED_SENTINEL": "keep-me",
        }
        with patch.dict(os.environ, inherited, clear=True):
            selected = launcher.runtime_environment(self.cuda)
        self.assertEqual(selected["NINFER_CUDA"], str(self.cuda))
        self.assertEqual(selected["LD_LIBRARY_PATH"],
                         str(self.cuda) + "/lib64:/usr/lib/wsl/lib:/opt/custom-library/lib")
        self.assertEqual(selected["UNCHANGED_SENTINEL"], "keep-me")

    def test_ldd_rejects_cuda_13_3_for_selected_13_4(self):
        foreign = "/usr/local/cuda-13.3/lib64/libcudart.so.13"
        for output in (
            "libcudart.so.13 => {} (0x00000000)\n".format(foreign),
            # ldd can print direct absolute-path DT_NEEDED / LD_PRELOAD entries.
            "{} (0x00000000)\n".format(foreign),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.evidence(output)

    def test_ldd_rejects_13_40_directory_with_13_4_name_prefix(self):
        output = "libcudart.so.13 => /usr/local/cuda-13.40/lib64/libcudart.so.13 (0x00000000)\n"
        with self.assertRaises(ValueError):
            self.evidence(output)

    def test_ldd_records_selected_13_4_runtime(self):
        runtime = "/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13"
        output = "libcudart.so.13 => {} (0x00000000)\n".format(runtime)
        evidence = self.evidence(output)
        self.assertEqual(evidence["exit_code"], 0)
        self.assertEqual(evidence["output"], output)
        self.assertEqual(evidence["resolved_cudart"], [{
            "library": "libcudart.so.13", "resolved_path": str(Path(runtime).resolve()),
        }])


if __name__ == "__main__":
    unittest.main(verbosity=2)
