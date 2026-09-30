"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def fake_sysfs(root: Path, nodes: list) -> None:
    """nodes: (gfx_target_version, simd_count, render_minor, product_name or None, vram_bytes)"""
    for i, (ver, simd, minor, name, vram) in enumerate(nodes):
        n = root / "class/kfd/kfd/topology/nodes" / str(i)
        n.mkdir(parents=True)
        (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\n"
                                      f"gfx_target_version {ver}\ndrm_render_minor {minor}\n")
        if simd:
            d = root / f"class/drm/renderD{minor}/device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_total").write_text(str(vram))
            if name is not None:
                (d / "product_name").write_text(name + "\n")


class KfdDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.win = setup.WIN
        setup.WIN = False

    def tearDown(self):
        setup.WIN = self.win
        self.tmp.cleanup()

    def test_every_family(self):
        fake_sysfs(self.root, [
            (0, 0, 0, None, 0),                                   # the CPU node: skipped
            (110001, 120, 128, "", 16 << 30),                     # gfx1101 without a product name
            (120000, 64, 129, None, 16 << 30),                    # gfx1200, no product_name file
            (120001, 128, 130, "AMD Radeon AI PRO R9700", 32 << 30),
            (110002, 64, 131, None, 8 << 30),                     # gfx1102: listed, not supported
            (100306, 4, 132, None, 512 << 20),                    # an integrated gfx1036: listed, not supported
            (110000, 192, 133, "Radeon RX 7900 XTX", 24 << 30),
        ])
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1036", "gfx1100"])
        self.assertEqual([x["index"] for x in g], [0, 1, 2, 3, 4, 5])          # HIP numbers: GPU nodes only
        self.assertEqual(g[0]["name"], setup.AMD_NAMES["gfx1101"])
        self.assertEqual(g[1]["name"], setup.AMD_NAMES["gfx1200"])
        self.assertEqual(g[2]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(g[3]["name"], "AMD Radeon (gfx1102)")
        self.assertAlmostEqual(g[2]["vram_gb"], 32.0)
        ok = [x["arch"] for x in g if setup.amd_problem(x) is None]
        self.assertEqual(ok, ["gfx1101", "gfx1200", "gfx1201", "gfx1100"])
        self.assertIn("gfx1102", setup.amd_problem(g[3]))
        self.assertIn("gfx1036", setup.amd_problem(g[4]))

    def test_no_kfd(self):
        self.assertEqual(setup.amd_gpus(str(self.root)), [])

    def test_rocm_index_per_family(self):
        for arch in setup.AMD_ARCHS:
            self.assertIn(arch, setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1101"].endswith("/gfx110X-dgpu/"))
        self.assertTrue(setup.ROCM_INDEXES["gfx1200"].endswith("/gfx120X-all/"))
        self.assertEqual(setup.ROCM_INDEXES["gfx1101"], setup.ROCM_INDEXES["gfx1100"])
        self.assertEqual(setup.ROCM_INDEXES["gfx1200"], setup.ROCM_INDEXES["gfx1201"])


if __name__ == "__main__":
    unittest.main()
