<h1 align="center">PolyStrata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run large AI models on your own PC</b><br>
Qwen3.8-Flash-Next and GLM 5.3 Flash · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden the model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

PolyStrata runs large AI models on a normal PC, models that usually need a server. They chat, write code and work
with your apps and coding agents. Nothing leaves your PC.

It runs two models, each with its own engine:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on an NVIDIA or AMD graphics card with
  12 GB or more, on Windows or Linux. It also reads pictures. Its engine is the Strata project's, unchanged.
- **[GLM 5.3 Flash](docs/GLM.md)** on Linux with one NVIDIA card. Its engine was written for PolyStrata. It is
  experimental: it has run on one PC, with a 32 GB card.

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

Measured for PolyStrata on one workstation, at a 32K context.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q, 120 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads a 3K prompt | Reads a 12K prompt | Reads a 29K prompt |
| --- | ---: | ---: | ---: | ---: |
| **3.0BIT** | 27 tokens/s | 373 tokens/s | 334 tokens/s | 312 tokens/s |

</td></tr>
</table>

llama.cpp with the same file on the same PC wrote 26 tokens/s and read about 265. GLM 5.3 Flash is a much larger
model than Qwen3.8-Flash-Next, and about a fifth of its experts fit in this card's 32 GB, so it writes more slowly.
What was measured and how: [docs/GLM.md](docs/GLM.md#what-it-was-measured-on).

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
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

### GLM 5.3 Flash

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** GeForce RTX 20 series card or newer. It has run on **32 GB of VRAM** only. About 12 GB are taken before any expert, so a 12 GB card cannot start it. |
| **RAM** | **96 GB or more** beside a 32 GB card. A smaller card needs more RAM ([how much](#which-model-should-i-pick)). It has run with 120 GB. |
| **Disk** | 114 GB for the model file, on an SSD. |
| **System** | Linux, and a current NVIDIA driver. |

The installer sets up everything else. It compiles the engine on your PC and offers to install what that needs
(g++ and the CUDA toolkit). Not yet: Windows, AMD cards, several cards, pictures.

The full list: [docs/GLM.md](docs/GLM.md#what-you-need).

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
- whether it should read pictures (Qwen3.8-Flash-Next).

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB for Qwen3.8-Flash-Next,
114 GB for GLM 5.3 Flash) and starts it. If the download stops, run it again: it continues where it left off. Your
browser opens the PolyStrata app at `http://127.0.0.1:8080`.

GLM 5.3 Flash is in the installer's list of models. `./setup.sh --family glm` goes straight to it
([what it asks and writes](docs/GLM.md#install)).

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

It comes in one size so far, **3.0BIT**: 3-bit experts, a 114 GB download. Your graphics card decides how much RAM
it needs, because the experts the card does not hold have to stay in RAM.

| Your graphics card | RAM it needs | |
| --- | --- | --- |
| **32 GB** | **96 GB** | the PC it was measured on, with 120 GB |
| **24 GB** | about 104 GB | not run yet |
| **16 GB** | about 113 GB | not run yet; the card holds 4% of the experts, so expect it to be slower |
| **12 GB or less** | | it does not start |

The RAM numbers are the installer's estimate from the one PC it has run on. With less RAM it still starts, but it
reads experts from the SSD while it answers and gets much slower. The installer says so and asks first.

Other quantizations of GLM 5.3 Flash are not supported yet: [what is missing](docs/GLM.md#what-is-missing).

## Using it

Both models are used the same way: the same app, the same API.

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

Where the two differ:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash |
| --- | --- | --- |
| **Pictures** | Say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app. AMD cards read pictures on Linux through the processor; on Windows they can't yet. | Not yet. |
| **Several requests** | One at a time by default, the others wait. `"parallel": 2` answers several at once ([BATCHING.md](docs/BATCHING.md)); on a 12 GB card each answer gets slower. | One at a time, the others wait. |
| **Several conversations** | The last one is kept, so follow-up messages start in seconds. Two engine arguments keep several in RAM ([details](docs/DETAILS.md#using-it)). | The last one is kept. Setup asks whether to keep several in RAM, so that an agent and its subagents take turns without their prompts being read again ([how](docs/GLM.md#several-conversations-at-once)). |
| **Long prompts** | The first message of a chat is read in full, about 1 minute per 30,000 tokens. | The first message is read in full, about 1.5 minutes per 30,000 tokens. |

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
12-32 GB. PolyStrata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
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

- **The model has 12,096 experts**, 288 in each of 42 layers. Each word needs 8 from every layer.
- **Your graphics card** keeps the ones used most often, about 2,570 of them on a 32 GB card. The engine counts
  which ones your work uses and chooses again at the next start.
- **While it writes**, your processor runs the experts the card does not hold. **While it reads a prompt**, they
  pass through the card, 1,024 tokens at a time.
- **The model file is used where it is**, with no second copy on the disk. A start takes about 6 seconds once the
  file is in your RAM.

Every part and its numbers: [docs/GLM.md](docs/GLM.md#how-it-uses-the-card).

## Credits and license

Qwen3.8-Flash-Next is by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
GLM 5.3 Flash runs from the 3-bit file by
[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF). PolyStrata is built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors, and uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). PolyStrata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support

PolyStrata is free and open source. Most of it is the work of [Strata](https://github.com/Niko1221/Strata)'s author. If
it is useful to you, you can support that work:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
