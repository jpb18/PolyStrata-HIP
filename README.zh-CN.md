<h1 align="center">PolyStrata</h1>

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>在你自己的电脑上运行大型 AI 模型</b><br>
Qwen3.8-Flash-Next 和 GLM 5.3 Flash · 免费开源</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="模型写出的体素宝塔花园，在浏览器中运行"></a><br>
<sub>体素宝塔花园，一次提示生成，在 RTX 5070 上用 <a href="https://github.com/Niko1221/Strata">Strata</a> 运行（Qwen3.8-Flash-Next IQ3_S，128K 上下文）·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">完整视频（49 秒）</a></sub></p>

PolyStrata 能在普通电脑上运行通常要用服务器才能跑的大型 AI 模型。它们能聊天、写代码，还能配合你的应用和编程智能体一起工作。
所有数据都留在你的电脑上。

它能跑两个模型，各有各的引擎：

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**：12 GB 及以上的 NVIDIA 或 AMD 显卡，Windows 或
  Linux。它还能看图片。引擎是 Strata 项目的，没有改动。
- **[GLM 5.3 Flash](docs/GLM.md)**：Linux 加一张 NVIDIA 显卡。引擎是为 PolyStrata 写的。目前是实验性的：只在一台电脑上跑过，
  显卡是 32 GB。

第二个引擎还能跑 Qwen3.6-35B-A3B、Ornith 1.5 和 Gemma 4 26B-A4B，文件 13 到 17 GB；服务器运行时可以在已安装的模型之间切换。
它们也在安装程序的模型列表里，想直接装其中一个，运行 `./setup.sh --family qwen36`（或 `ornith`、`gemma4`）。
测了什么见[模型和后端](docs/BACKENDS.md)（英文）。

PolyStrata 基于 Niko1221 的 [Strata](https://github.com/Niko1221/Strata)。

## 速度有多快？

一个 token 大约相当于 ¾ 个英文单词。

- **写回答：** 短对话中回复出现的速度。每秒 60 个 token 已经比你的阅读速度还快。
- **读提示：** 读入你发送内容的速度（文档、代码或聊天记录）。

### Qwen3.8-Flash-Next

由 Strata 项目在两台普通的游戏电脑上测得，提示长 32K token。

<table>
<tr><th>NVIDIA：RTX 5070（12 GB）、Ryzen 5 7600、64 GB 内存</th><th>AMD：RX 9070 XT（16 GB）、Ryzen 9 3900X、47 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA：Q2_0 用的是引擎 0.1.36，其他各行用的是 0.1.26（4K 回答，32K 提示）。完整表格见
[DETAILS.md](docs/DETAILS.md#speed-measured)。显存越大的显卡越快：RTX 3090（24 GB）写回答应该能达到每秒约
100-140 个 token。长对话和其他显卡的数据：[各规格的速度](docs/MODELS.md#how-fast-is-each-size)、
[社区测试结果](docs/COMMUNITY_BENCHMARKS.md)。

### GLM 5.3 Flash

由 PolyStrata 在一台工作站上测得，提示长 29K token。

<table>
<tr><th>NVIDIA：RTX 5090（32 GB）、Xeon Platinum 8470Q（25 vCPU）、120 GB 内存</th></tr>
<tr><td>

| 规格 | 写回答 | 读提示 |
| --- | ---: | ---: |
| **3.0BIT** | 27 tokens/s | 312 tokens/s |
| **REAP50-Q3_K_M** | 27 tokens/s | 384 tokens/s |

</td></tr>
</table>

同一台电脑上用 llama.cpp 跑 3.0BIT 的文件，写回答每秒 26 个 token，读提示每秒约 265 个。GLM 5.3 Flash 比
Qwen3.8-Flash-Next 大得多，这张卡的 32 GB 显存只放得下它约五分之一的专家，所以写得慢一些。
测了什么、怎么测的：[docs/GLM.md](docs/GLM.md#what-it-was-measured-on)。

## 你需要什么

### Qwen3.8-Flash-Next

| | |
| --- | --- |
| **显卡** | **NVIDIA** GeForce RTX 20、30、40 或 50 系列，或 **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700 或 RX 6800 / 6900 系列。需要 **12 GB 或以上显存**。 |
| **内存** | 32 GB 或以上。内存大小决定能装下[哪个规格](#该选哪个模型)。64 GB 可以运行所有规格。 |
| **硬盘** | 约 80 GB 可用空间。尽量用 SSD：第一次启动会快很多。 |
| **系统** | Windows 10 / 11 或 Linux，以及 NVIDIA 或 AMD 的最新显卡驱动。 |

其他的都由安装程序搞定。两到三张显卡可以分担同一个模型（[多 GPU](docs/MULTI_GPU.md)）。

实验性支持，由社区成员在自己的机器上编写和测试：

- **较老的显卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[较老的 GPU](docs/OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上从源码构建：[Intel Arc](docs/INTEL_ARC.md)。
- **不支持 AVX2 的老处理器**：能用，但很慢。[较老的 CPU](docs/INSTALL.md#older-cpus-experimental)。

完整列表：[docs/INSTALL.md](docs/INSTALL.md#what-you-need)。

### GLM 5.3 Flash

| | |
| --- | --- |
| **显卡** | 一张 **NVIDIA** GeForce RTX 20 系列或更新的显卡。目前只在 **32 GB 显存**上跑过。放专家之前就要占掉约 12 GB，所以 12 GB 的卡启动不了。 |
| **内存** | 配 32 GB 显卡时，完整模型需要 **96 GB 或以上**，专家减半的规格需要 60 GB。显卡越小，需要的内存越多（[具体多少](#该选哪个模型)）。实际跑过的是 120 GB。 |
| **硬盘** | 模型文件 114 GB 或 79 GB，要放在 SSD 上。 |
| **系统** | Linux，以及 NVIDIA 的最新显卡驱动。 |

其他的都由安装程序搞定。它会在你的电脑上编译引擎，缺 g++ 和 CUDA 工具包时会提出帮你装。
暂时不支持：Windows、AMD 显卡、多张显卡、图片。

完整列表：[docs/GLM.md](docs/GLM.md#what-you-need)。

## 安装

### 让你的 AI 帮你装

你在用 AI 编程助手吗（Claude Code、Cursor、Codex、GitHub Copilot 等）？把下面这段粘贴给它：

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

它会检查你的显卡、内存和硬盘，选出合适的模型。然后安装并启动它，再告诉你怎么连接你的应用。AI 工具也可以通过
PolyStrata 的 [MCP 服务器](docs/MCP_SERVER.md)来安装、启动和停止 PolyStrata。

### 或者自己动手

[下载 PolyStrata](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) 并解压（或者用 `git clone`）。
**Windows：** 双击 **`START-HERE.bat`**。**Linux：** 在 PolyStrata 文件夹里运行 **`./setup.sh`**。

安装程序会识别你的显卡，并装好对应的引擎。它会问你几个问题：

- 用哪个模型、哪个规格，
- 上下文多大（模型能记住多少文字），
- 是否要识别图片（GLM 5.3 Flash 不支持）。

每次直接按回车就是推荐选项。之后它会下载模型（Qwen3.8-Flash-Next 约 70 GB，GLM 5.3 Flash 79 或 114 GB）并启动。
如果下载中断了，再运行一次即可：它会从中断的地方继续。浏览器会打开 PolyStrata 应用，地址是 `http://127.0.0.1:8080`。

GLM 5.3 Flash 就在安装程序的模型列表里。想直接装它，运行 `./setup.sh --family glm`
（[它会问什么、写什么](docs/GLM.md#install)）。

> **模型启动时，你的电脑可能会变慢或卡住 1-3 分钟**（第一次最久）。
> PolyStrata 会把 35-55 GB 加载到内存，并为显卡锁定其中一部分。这是正常的。请耐心等待，不要关闭窗口。
> 窗口里会显示 PolyStrata 正在做什么。

**下次使用时**，再运行一次 `START-HERE.bat`（或 `./setup.sh`）。它会马上启动，已下载的东西不会重复下载。关闭它的窗口就能停止模型。
`UPDATE.bat`（`./update.sh`）只更新 PolyStrata，不启动。更新、Docker、多张显卡、文件存放位置以及所有选项：
[docs/INSTALL.md](docs/INSTALL.md)。

## 该选哪个模型

### Qwen3.8-Flash-Next

安装程序会根据你的内存推荐一个。同一个模型有几种规格，压缩程度不同。规格越小越快，越大越聪明一些。

| 你的内存 | 选择 | 原因 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB 装得下，而且专为代码打造（如果显卡是 24 GB，Q2_0 和 IQ2_XS 也能跑） |
| **48 GB** | **IQ2_XS**（或 Q2_0，最快） | 更大的规格装不下 |
| **64 GB** | **IQ2_XS**（推荐），或 IQ3_XXS / IQ3_S | 所有规格都装得下；IQ3_S 最好，也最慢 |
| **96 GB 或以上** | **IQ3_S**，或 Unsloth 的 UD-IQ4_XS（约 4-bit） | 开着其他程序也能放下最大的规格 |

- **[Coder](docs/MODELS.md#coder)：** 编程版本，去掉了一半专家。它达到完整模型 SWE-bench Verified 分数的 91%（由其作者测得），
  32 GB 内存就能装下。代码以外的能力较弱，包括中文和其他中日韩文本（#438）。这类用途请选 Q2_0、IQ2_XS 或 IQ3_S，
  它们保留了所有专家。
- **[Swift 1.5](docs/MODELS.md#swift-15)：** 一个微调版本，回答前思考的时间短得多。你能更早拿到答案，质量基本不变。
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)：** Unsloth 的约 4-bit 版本，质量介于 IQ3_S 和 UD-Q4_K_XL 之间。
  下载 94 GB。内存少于约 80 GB 时，PolyStrata 回答时要从 SSD 读取其中一部分，所以会更慢（NVMe SSD 有帮助）。
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)**（实验性）：最接近完整模型。但 PolyStrata
  回答时要从 SSD 读取其中大部分内容，所以在 64 GB 的电脑上每秒只能写 7-8.5 个 token。
- **[OrcaRouter 的 Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)：** 需要手动设置，不在安装程序的菜单里。

规格、下载以及各配置能装下什么：[docs/MODELS.md](docs/MODELS.md)。以后想再添加模型，运行
`SETUP.bat`（Linux：`./setup.sh --setup`）。

### GLM 5.3 Flash

目前有两个规格，安装程序会选内存够用的第一个。

| 规格 | 是什么 | 下载 | 配 32 GB 显卡需要的内存 |
| --- | --- | ---: | ---: |
| **3.0BIT** | 完整模型 | 114 GB | 96 GB |
| **REAP50-Q3_K_M** | 去掉了一半的路由专家（patrickbdevaney 的 [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)） | 79 GB | 60 GB |

需要多少内存取决于你的显卡，因为显卡放不下的专家都得留在内存里。

| 你的显卡 | 3.0BIT | REAP50-Q3_K_M | |
| --- | ---: | ---: | --- |
| **32 GB** | **96 GB** | **60 GB** | 跑过的就是这台电脑，内存 120 GB |
| **24 GB** | 约 104 GB | 约 68 GB | 还没跑过 |
| **16 GB** | 约 113 GB | 约 77 GB | 还没跑过 |
| **12 GB 或以下** | | | 启动不了 |

内存数字是安装程序根据唯一跑过的那台电脑估算的。内存不够也能启动，但回答时要从 SSD 读专家，会慢很多。
安装程序会先说明并征求你的同意。

GLM 5.3 Flash 的其他量化版本暂时不支持：[还缺什么](docs/GLM.md#what-is-missing)。

## 使用方法

两个模型的用法一样：同一个应用，同一套 API。

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="应用的 Monitor 标签页，旁边是一个编程智能体"><br>
<sub>应用的 <b>Monitor</b>（左），编程智能体正在写视频里的宝塔花园（右）。截图来自 <a href="https://github.com/Niko1221/Strata">Strata</a>。</sub></p>

- **在浏览器里：** 打开 `http://127.0.0.1:8080`。里面有 **Chat**（聊天）、实时 **Monitor**（监控模型和你的
  GPU/CPU/内存），以及 **About**（设置和地址）。
- **你的应用和编程智能体：** 添加一个“OpenAI 兼容”的提供商，base URL 填 **`http://127.0.0.1:8080/v1`**。
  API key 和模型名随便填都行。
  - 使用 Anthropic API 的应用：`http://127.0.0.1:8080/v1/messages`（Claude Code：
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）。
  - Codex CLI 和其他使用 OpenAI Responses API 的应用：`/v1/responses`
    （[设置方法](docs/DETAILS.md#the-responses-api-and-codex-cli)）。
- **思考：** 在聊天菜单或应用的“reasoning effort”里选择 **off、low、medium 或 high**。off 最快，
  high 最适合难题。
- **从手机或另一台电脑访问：** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`。一定要设置 key。

两个模型不一样的地方：

| | Qwen3.8-Flash-Next | GLM 5.3 Flash |
| --- | --- | --- |
| **图片** | 在安装时对“Images?”选是。然后在聊天里点 **Picture**，或在你的应用里附上图片。AMD 显卡在 Linux 上通过处理器识别图片；在 Windows 上暂时还不行。 | 暂时不支持。 |
| **多个请求** | 默认一次只回答一个，其他请求排队等待。设置 `"parallel": 2` 可以同时回答多个（[BATCHING.md](docs/BATCHING.md)）；在 12 GB 显卡上，这会让每个回答变慢。 | 一次只回答一个，其他请求排队等待。 |
| **多个对话** | 保留最近的一个对话，所以后续消息几秒内就开始回答。加两个引擎参数可以在内存里保留多个（[详情](docs/DETAILS.md#using-it)）。 | 保留最近的一个对话。安装程序会问要不要在内存里保留多个，这样智能体和子代理轮流请求时不用重读提示（[原理](docs/GLM.md#several-conversations-at-once)）。 |
| **长提示** | 对话的第一条消息要完整读入，大约每 30,000 个 token 需要 1 分钟。 | 第一条消息要完整读入，大约每 30,000 个 token 需要 1.5 分钟。 |

更多：[聊天记录存在哪里](docs/INSTALL.md#where-things-are-stored)、[API](docs/DETAILS.md#using-it)。

## 出问题了？

- **PolyStrata 第一次启动时电脑卡死了。** 加载模型时这是正常的。请耐心等待，不要关闭窗口。
  10 分钟后还卡着？重启电脑，关掉其他程序再试一次，或者换一个更小的规格。
- **下载或安装时中断了。** 再运行一次 `START-HERE.bat`（或 `./setup.sh`）。它会从中断的地方继续。
- **非常慢，硬盘灯一直闪，或者提示“the engine stopped unexpectedly”。** 你的电脑可用内存不够。
  关掉其他程序（浏览器很占内存），或者换一个更小的规格（Q2_0 或 IQ2_XS）。
- **提示 8080 端口已被占用。** PolyStrata 已经在运行了。找一下它的窗口。

更多问题和解决办法：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。还是解决不了？提交一个
[issue](https://github.com/VecSzn/PolyStrata/issues)，并附上 PolyStrata 文件夹里的 `strata-<model>.log`。发现了安全问题？
请私下报告：[SECURITY.md](SECURITY.md)。

## 它是怎么工作的？

这类模型通常运行在拥有数百 GB 显存的服务器上。而你的显卡只有 12-32 GB。PolyStrata 让模型装得下的办法是
**把工作分摊到整台电脑上**。可以想象一个厨房：常用的东西放在台面上，其余的放在储藏室里。

### Qwen3.8-Flash-Next

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="模型的 24,576 个专家：最常用的在显卡上，全部在内存里，一张查找表在 SSD 上"></p>

- **这个模型是由 24,576 个小专家（“experts”）组成的团队。** 每个词只需要其中 10 个。
- **你的显卡** 存放最常用的几千个专家。**你的内存** 存放全部专家，
  **你的处理器** 同时处理其余的部分。**你的 SSD** 存放一张大查找表。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="一个小助手猜测接下来的几个词；大模型一次性全部检查，保留正确的"></p>

- **先猜，再检查：** 一个小助手先猜接下来的几个词，大模型再一次性检查它们。答案完全一样，但快 1.6-1.8 倍。
- **长文本分大块读入**（每次最多 8,192 个 token），速度超过每秒 1,000 个 token。

更详细的解释：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。每个部分及其数据：
[详细说明](docs/DETAILS.md#how-it-works)和[论文](docs/paper/Strata-Paper.pdf)。

### GLM 5.3 Flash

- **这个模型有 12,096 个专家**，42 层，每层 288 个。每个词要从每一层里用 8 个。
- **你的显卡** 存放最常用的那些，32 GB 的卡上约 2,570 个。引擎会统计你的使用中哪些专家用得多，下次启动时重新挑选。
- **写回答时**，显卡放不下的专家由处理器来算。**读提示时**，它们分批经过显卡，每批 1,024 个 token。
- **模型文件原地使用**，硬盘上不会多出一份副本。文件进了内存缓存之后，启动只要 6 秒左右。

每个部分及其数据：[docs/GLM.md](docs/GLM.md#how-it-uses-the-card)。

## 致谢与许可证

Qwen3.8-Flash-Next 来自 [Qwen 团队](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，由
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）
和 Unsloth 压缩。GLM 5.3 Flash 用的是
[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) 和 patrickbdevaney 的文件。PolyStrata 基于
Niko1221 和各位贡献者的 [Strata](https://github.com/Niko1221/Strata)，并使用了
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 的部分代码。完整致谢：
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)。PolyStrata 以 [MIT 许可证](LICENSE)开源。少数部分和每个模型
有各自的许可证（[具体是哪些](docs/HOW_IT_WORKS.md#license)）。

## 支持

PolyStrata 免费且开源，其中大部分是 [Strata](https://github.com/Niko1221/Strata) 作者的成果。如果它对你有用，欢迎支持上游的开发：

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="请我喝杯咖啡" height="50"></a></p>
