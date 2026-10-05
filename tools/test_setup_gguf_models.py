"""Tests for setup.py's Qwen3.6-35B-A3B, Ornith 1.5 and Gemma 4 26B-A4B (docs/BACKENDS.md): the pinned files, the
estimates for a small card, and that each is installed by GLM's steps with nothing of GLM's own (no shipped expert
profile, no draft block to fetch).  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_gguf_models
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402   (the module: its test classes are not this file's)

# family: (size, file, repository, bytes)
NEW = {
    "qwen36": ("UD-Q3_K_M", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", 17104402720),
    "ornith": ("APEX-I-COMPACT", "Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf",
               "mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF", 17437861408),
    "gemma4": ("UD-Q3_K_M", "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", "unsloth/gemma-4-26B-A4B-it-GGUF", 12728497888),
}


class Pins(unittest.TestCase):
    def test_the_files(self):
        for family, (size, file, repo, n) in NEW.items():
            fam = setup.FAMILIES[family]
            self.assertEqual(fam["engine"], "glm", family)
            self.assertRegex(fam["hf"], rf"^https://huggingface.co/{repo}/resolve/[0-9a-f]{{40}}/$")
            self.assertEqual(list(setup.family_models(family)), [size])
            self.assertEqual((setup.model_file(fam, size, 1), setup.model_shards(fam, size)), (file, 1))
            self.assertEqual(list(fam["sha256"]), [file])
            self.assertEqual(fam["sha256"][file][0], n)
            self.assertRegex(fam["sha256"][file][1], r"^[0-9a-f]{64}$")
            d = fam["models"][size]
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])            # the experts are part of the file

    def test_two_families_have_a_size_of_one_name(self):
        q, g = setup.FAMILIES["qwen36"]["models"]["UD-Q3_K_M"], setup.FAMILIES["gemma4"]["models"]["UD-Q3_K_M"]
        self.assertNotEqual(q["download_gb"], g["download_gb"])
        self.assertNotIn("UD-Q3_K_M", setup.MODELS)                      # the other families' sizes are not theirs
        for f in ("qwen", "swift", "coder", "unsloth", "glm"):
            self.assertNotIn("UD-Q3_K_M", setup.family_models(f), f)

    def test_tags_tell_the_families_apart(self):
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        for a in tags:
            self.assertEqual([b for b in tags if b.startswith(a)], [a])


class Estimates(unittest.TestCase):
    def test_a_32_gb_card_holds_every_expert(self):
        for family in NEW:
            for d in setup.family_models(family).values():
                self.assertEqual(setup.glm_held_gb(d, 31.84), d["arena_gb"], family)
                self.assertEqual(setup.glm_ram_need(d, 31.84), setup.GLM_RAM_BESIDE_GB)

    def test_an_8_gb_card(self):
        # what the families' menu lines say of an 8 GB card
        need = {f: setup.glm_ram_need(next(iter(setup.family_models(f).values())), 8.0) for f in NEW}
        self.assertAlmostEqual(need["qwen36"], 21, delta=0.5)
        self.assertAlmostEqual(need["ornith"], 22, delta=0.5)
        self.assertAlmostEqual(need["gemma4"], 17, delta=0.5)
        for f, n in need.items():
            self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", setup.FAMILIES[f]["about"])
            self.assertEqual(next(iter(setup.family_models(f).values()))["ram_gb"], round(n))

    def test_check_lines(self):
        d = setup.FAMILIES["qwen36"]["models"]["UD-Q3_K_M"]
        self.assertIn("fits by setup's estimate (~21 GB of RAM", setup.glm_verdict(d, 32, 8.0, False))
        self.assertIn("read from the SSD", setup.glm_verdict(d, 12, 8.0, False))
        self.assertIn("does not fit", setup.glm_verdict(d, 64, 6.0, False))
        self.assertIn("AMD", setup.glm_verdict(d, 64, 24.0, True))


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for family, (size, *_rest) in NEW.items():
                p = Path(t) / f"strata-{setup.FAMILIES[family]['tag']}{size.lower()}.json"
                p.write_text(json.dumps({"exe": "/x/engine-glm/" + setup.GEXE, "args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"], ch["context"]), (family, size, 8192))


class Main(unittest.TestCase):
    """setup.main() for the three families, every outside effect mocked."""
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install(self):
        for family, (size, file, _repo, _n) in NEW.items():
            with self.subTest(family):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, cfg = self.main(family=family, ram=32.0, vram=8.0)
                self.assertEqual(code, 0, out)
                fam = setup.FAMILIES[family]
                self.assertEqual(self.downloads, [fam["hf"] + file])
                self.assertEqual(self.verified, [(file, *fam["sha256"][file])])
                tag = fam["tag"] + size
                pack = self.t / "data" / "packs" / tag.lower()
                # the tokenizer alone: no draft block is fetched for these
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                self.assertEqual(self.runs[0][self.runs[0].index("--out") + 1], str(pack))
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                self.assertFalse((pack / "expert-profile.txt").exists())   # none is shipped: the engine starts it
                self.assertNotIn("--draft-model", args)
                self.assertEqual(cfg["model_name"], f"{fam['name']}-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"{fam['title']} {size}")
                self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_GLM_DIR / setup.GEXE)
                self.assertIn("% of its experts fit in VRAM (100% there)", out)

    def test_a_pc_below_the_qwen_models_ram_is_not_stopped(self):
        # 24 GB is under what the smallest Qwen3.8 size needs, and enough for these by their own estimate
        code, out, cfg = self.main(family="gemma4", ram=24.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("the Coder", out)
        self.assertIn("RAM: 24 GB", out)
        self.assertIsNotNone(cfg)

    def test_too_little_ram_for_the_model_stops(self):
        code, out, cfg = self.main(family="qwen36", ram=12.0, vram=8.0)
        self.assertNotEqual(code, 0)
        self.assertIn("wants about 21 GB", out)
        self.assertIsNone(cfg)
        code, out, cfg = self.main(family="qwen36", ram=12.0, vram=8.0, model="UD-Q3_K_M")   # named: a risk taken
        self.assertEqual(code, 0, out)
        self.assertIn("as you chose", out)

    def test_a_size_of_another_family_is_refused_by_name(self):
        code, out, cfg = self.main(family="gemma4", model="APEX-I-COMPACT")
        self.assertNotEqual(code, 0)
        self.assertIn("Gemma 4 26B-A4B has no APEX-I-COMPACT model file", out)
        self.assertIn("UD-Q3_K_M", out)
        code, out, cfg = self.main(family="qwen", model="UD-Q3_K_M")     # and the other way: who has it
        self.assertNotEqual(code, 0)
        self.assertIn("Qwen3.8-Flash-Next has no UD-Q3_K_M model file", out)
        self.assertIn("--family qwen36 --model UD-Q3_K_M, --family gemma4 --model UD-Q3_K_M", out)

    def test_the_context_note_is_the_familys(self):
        code, out, cfg = self.main(family="gemma4", ram=32.0, vram=8.0, answers={"Context?": "3"})
        self.assertEqual(code, 0, out)
        self.assertIn("0.9 GB more VRAM", out)                           # 65536: 32768 tokens of 28672 bytes more
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")


if __name__ == "__main__":
    unittest.main()
