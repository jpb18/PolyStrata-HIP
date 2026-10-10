<h1 align="center">PolyStrata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run large AI models on your own PC</b><br>
Qwen3.8-Flash-Next, GLM 5.3 Flash, Qwen3.6, Ornith 1.5 and Gemma 4 · free and open source</p>

<p align="center"><img src="docs/media/speed.svg" width="820" alt="Tokens per second while writing an answer: 73 to 75 for Qwen3.6-35B-A3B on a laptop with an 8 GB card, 332 to 334 on an RTX 5090"></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden the model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

PolyStrata runs large AI models on a normal PC, models that usually need a server. They chat, write code and work
with your apps and coding agents. Nothing leaves your PC.

It runs five models, on two engines:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on an NVIDIA or AMD graphics card with
  12 GB or more, on Windows or Linux. It also reads pictures. Its engine is the Strata project's, unchanged.

The other four run on an engine written for PolyStrata, with one NVIDIA card: on Linux, and the three smaller ones on
Windows too. They are experimental: they have run on one PC with a 32 GB card, and the three smaller ones on a laptop
with an 8 GB card as well.

- **[GLM 5.3 Flash](docs/GLM.md)**, the largest: a file of 79 to 114 GB, most of which stays in RAM.
- **[Qwen3.6-35B-A3B](docs/BACKENDS.md)**, a file of 13 to 23 GB. It also reads pictures.
- **[Ornith 1.5](docs/BACKENDS.md)**, which has Qwen3.6-35B-A3B's architecture and other weights: a file of 14 to 24 GB. It
  also reads pictures.
- **[Gemma 4 26B-A4B](docs/BACKENDS.md)**, a file of 11 to 17 GB. It also reads pictures.

The server can change between installed models while it runs ([how](docs/BACKENDS.md#changing-the-model)).

PolyStrata is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221.

## How fast is it?

A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (a document, code or a chat history).

### Qwen3.8-Flash-Next

Measured by the Strata project on two ordinary gaming PCs, with a 32K-token prompt.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each size](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

### GLM 5.3 Flash

Measured for PolyStrata on one workstation, with a short chat and a 28K-token prompt.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **3.0BIT** | 57-59 tokens/s | 784-785 tokens/s |
| **UD-IQ2_XXS** | 40-44 tokens/s | 1,020 tokens/s |
| **REAP50-Q3_K_M** | 56-62 tokens/s | 1,250 tokens/s |

</td></tr>
</table>

llama.cpp with the 3.0BIT file on the same PC wrote 26 tokens/s and read about 265. GLM 5.3 Flash is a much larger
model than the others, and about a fifth of its experts fit in this card's 32 GB, so it writes more slowly.
What was measured and how: [docs/GLM.md](docs/GLM.md#what-it-was-measured-on).

### Qwen3.6-35B-A3B

Measured for PolyStrata on two PCs, with a short chat and a prompt of 30,000 to 32,000 tokens: on the left a laptop
whose card has 8 GB, where the processor runs the experts the card has no room for, and on the right a workstation
whose card holds the whole model.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 73-75 tokens/s | 1,390 tokens/s |
| **UD-Q2_K_XL** | 91 tokens/s | 1,320 tokens/s |
| **UD-Q4_K_M** | 58-61 tokens/s | 1,100-1,350 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 332-334 tokens/s | 5,520-5,580 tokens/s |
| **UD-Q2_K_XL** | 368-369 tokens/s | 5,540-5,580 tokens/s |
| **UD-Q4_K_M** | 333-337 tokens/s | 5,630-5,660 tokens/s |

</td></tr>
</table>

A shorter prompt is read faster: one of 10,000 tokens at 1,450 tokens/s on the laptop and at 5,500 on the workstation.
llama.cpp with the whole model on the workstation's card wrote about 180 tokens/s and read prompts at 6,000 to 9,000.
It needs a card the model fits. The workstation's card held to 8 GB, a stand-in from before the laptop was measured,
is in [docs/BACKENDS.md](docs/BACKENDS.md#with-8-gb-of-vram).

### Ornith 1.5

Measured the same way, on the same two PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 70 tokens/s | 1,340 tokens/s |
| **APEX-I-MINI** | 63-68 tokens/s | 1,300-1,310 tokens/s |
| **APEX-I-QUALITY** | 42-48 tokens/s | 1,290-1,320 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 267-272 tokens/s | 5,700-5,890 tokens/s |
| **APEX-I-MINI** | 265-279 tokens/s | 5,450-5,510 tokens/s |
| **APEX-I-QUALITY** | 263-265 tokens/s | 5,600-5,610 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,380 tokens/s on the laptop and at 5,500 on the workstation. llama.cpp with the
whole model on the workstation's card wrote about 180 tokens/s and read prompts at 6,000 to 9,000.

### Gemma 4 26B-A4B

Measured the same way, on the same two PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 70-71 tokens/s | 1,910 tokens/s |
| **UD-Q2_K_XL** | 86-87 tokens/s | 1,840 tokens/s |
| **UD-Q4_K_M** | 67 tokens/s | 1,840 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 306 tokens/s | 7,170-7,210 tokens/s |
| **UD-Q2_K_XL** | 315 tokens/s | 6,920 tokens/s |
| **UD-Q4_K_M** | 301 tokens/s | 7,120-7,210 tokens/s |

</td></tr>
</table>

A prompt of 10,000 tokens is read at 1,950 tokens/s on the laptop and at 5,500 on the workstation. llama.cpp with the
whole model on the workstation's card wrote about 165 tokens/s and read prompts at 6,000 to 9,000.

The four models of PolyStrata's own engine were measured with its benchmark, which you can run on your PC:
`python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([what it sends and prints](docs/BACKENDS.md#benchmark)).

## What you need

### Qwen3.8-Flash-Next

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which size](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

### GLM 5.3 Flash

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer. It has run on **32 GB of VRAM** only. About 12 GB are taken before any expert, so a 12 GB card cannot start it. |
| **RAM** | Beside a 32 GB card, **96 GB or more** for the whole model, 82 GB for its 2-bit size, 60 GB for the size with half the experts. A smaller card needs more RAM ([how much](#which-model-should-i-pick)). It has run with 120 GB. |
| **Disk** | 114 GB, 102 GB or 79 GB for the model file, on an SSD. |
| **System** | Linux, and a current NVIDIA driver. |

The installer sets up everything else. It compiles the engine on your PC and offers to install what that needs
(g++ and the CUDA toolkit). Not yet: Windows, AMD cards, several cards, pictures.

The full list: [docs/GLM.md](docs/GLM.md#what-you-need).

### Qwen3.6-35B-A3B

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 5 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. |
| **RAM** | About **17, 21 or 27 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 24 GB holds the two smaller sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 13, 17 or 23 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it compiles the engine with Visual
Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not yet: AMD cards, several cards.

### Ornith 1.5

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 4 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. |
| **RAM** | About **18, 22 or 28 GB** beside an 8 GB card for the three sizes, the installer's estimate. A card with 24 GB holds the two smaller sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 14, 17 or 24 GB for the model file, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it compiles the engine with Visual
Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not yet: AMD cards, several cards.

### Gemma 4 26B-A4B

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer with **8 GB of VRAM or more**. About 6 GB are taken before any expert. It has run on a 32 GB card, on that card held to 8 GB, and on a laptop's 8 GB card. |
| **RAM** | About **15, 18 or 21 GB** beside an 8 GB card for the three sizes, the installer's estimate. By the same estimate a card with 16 GB holds the two smaller sizes whole. All three sizes have run on a laptop with 32 GB. |
| **Disk** | 11, 13 or 17 GB for the model file and its draft model, 1 GB more to read pictures. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

The installer sets up everything else, as it does for GLM 5.3 Flash. On Windows it compiles the engine with Visual
Studio's 2022 compiler and the CUDA toolkit, and offers to install them. Not yet: AMD cards, several cards.

What each of the three needs and how it was checked: [docs/BACKENDS.md](docs/BACKENDS.md#install).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop PolyStrata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download PolyStrata](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the PolyStrata folder.

The installer finds your card and sets up the right engine for it. It asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures (GLM 5.3 Flash reads none).

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB for Qwen3.8-Flash-Next,
79 to 114 GB for GLM 5.3 Flash, 11 to 24 GB for the other three) and starts it. If the download stops, run it again:
it continues where it left off. Your browser opens the PolyStrata app at `http://127.0.0.1:8080`.

All five are in the installer's list of models. `./setup.sh --family glm`, `qwen36`, `ornith` or `gemma4` goes
straight to one ([what it asks and writes](docs/GLM.md#install)).

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> PolyStrata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what PolyStrata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates PolyStrata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

### Qwen3.8-Flash-Next

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, PolyStrata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But PolyStrata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

### GLM 5.3 Flash

Three sizes so far. The installer takes the first one your RAM is enough for.

| Size | What it is | Download | RAM beside a 32 GB card |
| --- | --- | ---: | ---: |
| **3.0BIT** | the whole model, at about 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | the whole model, at about 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): it reads a prompt faster and writes more slowly | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | half of the routed experts removed ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) by patrickbdevaney) | 79 GB | 60 GB |

Your graphics card decides how much RAM it needs, because the experts the card does not hold have to stay in RAM.

| Your graphics card | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | the PC it has run on (120 GB of RAM) |
| **24 GB** | about 104 GB | about 90 GB | about 68 GB | not run yet |
| **16 GB** | about 113 GB | about 99 GB | about 77 GB | not run yet |
| **12 GB or less** | | | | it does not start |

The RAM numbers are the installer's estimate from the one PC it has run on. With less RAM it still starts, but it
reads experts from the SSD while it answers and gets much slower. The installer says so and asks first.

Other quantizations of GLM 5.3 Flash are not supported yet: [what is missing](docs/GLM.md#what-is-missing).

### Qwen3.6-35B-A3B

Three sizes so far. The installer takes UD-Q3_K_M, or UD-Q2_K_XL when the PC has too little RAM for it. UD-Q4_K_M is chosen by name: `--model UD-Q4_K_M`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | the whole model with 3-bit experts ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | about 21 GB |
| **UD-Q2_K_XL** | the same with 2- and 3-bit experts: the least RAM | 13 GB | about 17 GB |
| **UD-Q4_K_M** | the same with 4-bit experts | 23 GB | about 27 GB |

A larger card holds more of the model and needs less RAM: with UD-Q3_K_M about 17 GB beside a 12 GB card, 13 GB beside a 16 GB
card, and 10 GB once the card holds all of it. These are the installer's estimates. They have not been checked on
such a PC.

### Ornith 1.5

Three sizes so far. The installer takes APEX-I-COMPACT, or APEX-I-MINI when the PC has too little RAM for it. APEX-I-QUALITY is chosen by name: `--model APEX-I-QUALITY`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | the whole model with 3- and 4-bit experts ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) by mudler) | 17 GB | about 22 GB |
| **APEX-I-MINI** | the same with 2- and 3-bit experts: the least RAM | 14 GB | about 18 GB |
| **APEX-I-QUALITY** | the same with 4- to 6-bit experts | 24 GB | about 28 GB |

A larger card holds more of the model and needs less RAM: with APEX-I-COMPACT about 17 GB beside a 12 GB card, 13 GB beside a 16 GB
card, and 10 GB once the card holds all of it. These are the installer's estimates. They have not been checked on
such a PC.

### Gemma 4 26B-A4B

Three sizes so far. The installer takes UD-Q3_K_M, or UD-Q2_K_XL when the PC has too little RAM for it. UD-Q4_K_M is chosen by name: `--model UD-Q4_K_M`.

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | the whole model with 3-bit experts ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), and Google's draft model for it | 13 GB | about 18 GB |
| **UD-Q2_K_XL** | the same with 2-bit experts: the least RAM | 11 GB | about 15 GB |
| **UD-Q4_K_M** | the same with 4- and 5-bit experts | 17 GB | about 21 GB |

A larger card holds more of the model and needs less RAM: with UD-Q3_K_M about 14 GB beside a 12 GB card, and 10 GB once the card
holds all of it. These are the installer's estimates. They have not been checked on such a PC.

## Using it

All five models are used the same way: the same app, the same API.

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The app's Monitor tab next to a coding agent"><br>
<sub>The app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right). A screenshot from <a href="https://github.com/Niko1221/Strata">Strata</a>.</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.

Where they differ:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4 26B-A4B |
| --- | --- | --- | --- |
| **Pictures** | Say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app. AMD cards read pictures on Linux through the processor; on Windows they can't yet. | Not yet. | Say yes to "Images?" in setup, then as with Qwen3.8-Flash-Next. On a card with less than 16 GB the picture encoder runs on the processor, and a picture takes a few seconds. |
| **Several requests** | One at a time by default, the others wait. `"parallel": 2` answers several at once ([BATCHING.md](docs/BATCHING.md)); on a 12 GB card each answer gets slower. | One at a time, the others wait. | One at a time, the others wait. |
| **Several conversations** | The last one is kept, so follow-up messages start in seconds. Two engine arguments keep several in RAM ([details](docs/DETAILS.md#using-it)). | The last one is kept. Setup asks whether to keep several in RAM, so that an agent and its subagents take turns without their prompts being read again ([how](docs/GLM.md#several-conversations-at-once)). | As GLM 5.3 Flash. |
| **Long prompts** | The first message of a chat is read in full, about 1 minute per 30,000 tokens. | The first message is read in full, about 40 seconds per 30,000 tokens. | The first message is read in full: 30,000 tokens take 6 to 8 seconds on the workstation and 16 to 22 seconds on the 8 GB laptop. |

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time PolyStrata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** PolyStrata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/VecSzn/PolyStrata/issues) and attach `strata-<model>.log` from the PolyStrata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like these usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
8-32 GB. PolyStrata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

### Qwen3.8-Flash-Next

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

### GLM 5.3 Flash

<p align="center"><img src="docs/media/how-it-works-glm.svg" width="860" alt="The model's 12,096 experts: the most used on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 12,096 small experts**, 288 in each of 42 layers. Each word needs only 8 from every layer.
- **Your graphics card** keeps the ones used most often, about 2,470 of them on a 32 GB card: the engine counts which
  ones your work uses and chooses again at the next start. **Your RAM** holds the rest, and **your processor** runs
  them at the same time. **Your SSD** holds the model's file, which is used where it is, with no second copy; a start
  takes about 6 seconds once the file is in your RAM.

<p align="center"><img src="docs/media/guess-and-check-glm.svg" width="860" alt="A block of the model guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a block of the model made for guessing proposes the next few words. The model checks them all
  at once. You get the same answer, 1.1-1.2x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time, the experts the card does not hold passing through
  it), at 780 to 1,250 tokens per second.

Every part and its numbers: [docs/GLM.md](docs/GLM.md#how-it-uses-the-card).

### Qwen3.6-35B-A3B

<p align="center"><img src="docs/media/how-it-works-qwen36.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 10,240 small experts**, 256 in each of 40 layers. Each word needs only 8 from every layer.
- **Your graphics card** keeps as many as fit: about 2,000 of them on an 8 GB card, all of them from 24 GB on. While
  it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it holds serve
  about 55% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at the same time.
  **Your SSD** holds the model's file, which is used where it is, with no second copy.

<p align="center"><img src="docs/media/guess-and-check-qwen36.svg" width="860" alt="The model's draft block guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** the model's own draft block, which is in its file, proposes the next few words. The model
  checks them all at once. You get the same answer, 1.2-1.5x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works).

### Ornith 1.5

<p align="center"><img src="docs/media/how-it-works-ornith.svg" width="860" alt="The model's 10,240 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **Qwen3.6-35B-A3B's architecture with other weights:** a team of 10,240 small experts, 256 in each of 40 layers, and
  only 8 of every layer for each word.
- **Your graphics card** keeps as many as fit: about 2,500 of them on an 8 GB card, all of them from 24 GB on. While
  it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB card the ones it holds serve
  about 60% of what an answer uses. **Your RAM** holds the rest, and **your processor** runs them at the same time.
  **Your SSD** holds the model's file, which is used where it is, with no second copy.

<p align="center"><img src="docs/media/guess-and-check-ornith.svg" width="860" alt="The model's draft block guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** its draft block is in its file too, and proposes the next few words. It guesses right less
  often than Qwen3.6-35B-A3B's, so you get the same answer 1.1-1.2x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works).

### Gemma 4 26B-A4B

<p align="center"><img src="docs/media/how-it-works-gemma4.svg" width="860" alt="The model's 3,840 experts: the ones an answer keeps using on the graphics card, the rest in RAM, the model's file on the SSD"></p>

- **The model is a team of 3,840 small experts**, 128 in each of 30 layers. Each word needs only 8 from every layer,
  beside a part of the layer that every word uses.
- **Your graphics card** keeps as many as fit: about 640 of them on an 8 GB card, all of them from 16 GB on by the
  installer's estimate. While it writes, the engine brings in the experts the answer keeps choosing, and on the 8 GB
  card the ones it holds serve about 55% of what an answer uses. **Your RAM** holds the rest, and **your processor**
  runs them at the same time. **Your SSD** holds the model's file and its draft model, which are used where they are,
  with no second copy.

<p align="center"><img src="docs/media/guess-and-check-gemma4.svg" width="860" alt="Google's small draft model guesses the next words; the model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** Google's small draft model for it, a file of 0.5 GB that the installer downloads with the
  model, proposes the next few words. The model checks them all at once. You get the same answer, 1.3-1.6x sooner.
- **Long texts are read in big pieces** (1,024 tokens at a time). When the card does not hold every expert, the others
  pass through it once for the whole prompt.

Every part and its numbers: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works).

## Credits and license

Qwen3.8-Flash-Next is by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
GLM 5.3 Flash runs from the files of
[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF), Unsloth and patrickbdevaney.
Qwen3.6-35B-A3B is by the Qwen team and Gemma 4 by Google, both in
[Unsloth](https://huggingface.co/unsloth)'s quantization. Ornith 1.5 is by ornith-ai, in
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)'s APEX quantization.
PolyStrata is built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors, and uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). PolyStrata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).
