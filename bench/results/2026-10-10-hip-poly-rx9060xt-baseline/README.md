# strata-poly with HIP on an RX 9060 XT: the state the second phase starts from (2026-10-10)

The measurements of the port's first phase ([AMD_PORT_IMPLEMENTATION.md](../../../AMD_PORT_IMPLEMENTATION.md)), kept
here so the second phase ([AMD_PORT_PHASE2.md](../../../AMD_PORT_PHASE2.md)) can be held against them.

- Branch `hip-poly` at `ba706504`; `build-hip-poly-win/strata-poly.exe` built from it (SHA-256 starts `c19987a43c8f8a58`).
- RX 9060 XT 16 GB (gfx1200, Adrenalin 32.0.31041.1004), Ryzen 5 5600X, 31.9 GB RAM, Windows 11 Pro, ROCm
  10.2.0a20260930 (TheRock). Other programs open (browser, chat apps): about 13 GB of RAM in use beside the engine.
- Model: `unsloth/Qwen3.5-35B-A3B-GGUF` `Qwen3.5-35B-A3B-Q3_K_M.gguf`, SHA-256 `5607c8fc...ab1b2d` (experts IQ3_XXS
  gate/up and IQ4_XS down, dense Q8_0, output Q6_K, no MTP block).

| File | What |
| --- | --- |
| `server-benchmark-48k.json` | `polystrata.py benchmark --prompt-tokens 2000,8000,24000,34000`, context 49,152, 6 threads, after the phase's fixes |
| `server-benchmark-before-fixes.json` | the same benchmark (2000, 8000) before the expert-cache reserve and the hipBLAS warm-up |
| `parity-window1.json` | `tools/hip/poly_llama_parity.py --engine-args "--window 1 --threads 6"` after the fixes: 4 prompts, 64/64 tokens, largest difference 0.141 |
| `llama-hip-refs.json` | the reference those runs compared against: llama.cpp HIP (Unsloth's build b11160, `--cpu-moe`), greedy, 64 tokens, top 5 a step |

The figures, one run each with the machine in normal use:

| Prompt | Tokens | TTFT s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| short | 21 | 0.25 | 84 | 58.7 |
| long-2000 | 2,743 | 2.49 | 1,105 | 62.7 |
| long-8000 | 10,615 | 6.13 | 1,401 | 60.5 |
| long-24000 | 31,352 | 17.8 | 1,190 | 52.7 |
| long-34000 | 44,419 | 14.2 | 966 | 50.7 |

The long prompts of that benchmark start with the same text, so part of each was read already (the conversation
cache): its prefill counts the tokens read, not the prompt's.
