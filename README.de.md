<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Große KI-Modelle auf deinem eigenen PC</b><br>
Qwen3.8-Flash-Next, GLM 5.3 Flash, Qwen3.6, Ornith 1.5 und Gemma 4 · kostenlos und Open Source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Ein Voxel-Pagodengarten, den das Modell in PolyStrata geschrieben hat, läuft im Browser"></a><br>
<sub>Ein Voxel-Pagodengarten aus einem einzigen Prompt, auf einer RTX 5070 mit <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, 128K Kontext) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">ganzes Video (49 s)</a></sub></p>

PolyStrata lässt große KI-Modelle auf einem normalen PC laufen, Modelle, die sonst meist einen Server brauchen.
Sie chatten, schreiben Code und arbeiten mit deinen Apps und Coding-Agents zusammen. Nichts verlässt deinen PC.

Es lässt fünf Modelle laufen, auf zwei Engines:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** auf einer NVIDIA- oder AMD-Grafikkarte
  ab 12 GB, unter Windows oder Linux. Es versteht auch Bilder. Seine Engine ist die von
  [Strata](https://github.com/Niko1221/Strata), unverändert.

Die anderen vier laufen auf einer Engine, die für PolyStrata geschrieben wurde, mit einer NVIDIA-Karte: alle vier
unter Linux, die drei kleineren auch unter Windows. Sie sind experimentell: Sie sind bisher auf einem einzigen PC mit
einer 32-GB-Karte gelaufen, die drei kleineren außerdem auf einem Laptop mit einer 8-GB-Karte.

- **[GLM 5.3 Flash](docs/GLM.md)**, das größte: eine Datei von 79 bis 114 GB, die zum größten Teil im RAM bleibt.
- **[Qwen3.6-35B-A3B](docs/BACKENDS.md)**, eine Datei von 13 bis 23 GB. Es versteht auch Bilder.
- **[Ornith 1.5](docs/BACKENDS.md)**, mit der Architektur von Qwen3.6-35B-A3B und anderen Gewichten: eine Datei von
  14 bis 24 GB. Es versteht auch Bilder.
- **[Gemma 4 26B-A4B](docs/BACKENDS.md)**, eine Datei von 11 bis 17 GB. Es versteht auch Bilder.

Der Server kann im laufenden Betrieb zwischen installierten Modellen wechseln
([wie](docs/BACKENDS.md#changing-the-model), englisch).

PolyStrata baut auf [Strata](https://github.com/Niko1221/Strata) von Niko1221 auf.

## Wie schnell ist es?

Ein Token ist etwa ¾ eines Wortes.

- **Schreibt Antworten:** wie schnell die Antwort in einem kurzen Chat erscheint. 60 Tokens pro Sekunde sind
  schneller, als du lesen kannst.
- **Liest deinen Prompt:** wie schnell es aufnimmt, was du schickst (hier ein Dokument, Code oder Chatverlauf mit
  32K Tokens).

### Qwen3.8-Flash-Next

Gemessen von [Strata](https://github.com/Niko1221/Strata) auf zwei ganz normalen Gaming-PCs, mit einem Prompt von 32K Tokens.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 Tokens/s | 2,650 Tokens/s |
| **IQ2_XS** | 79 Tokens/s | 2,090 Tokens/s |
| **IQ3_XXS** | 62 Tokens/s | 1,750 Tokens/s |
| **IQ3_S** | 53 Tokens/s | 1,620 Tokens/s |
| **Coder** | 55 Tokens/s | 2,180 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 Tokens/s | 1,160 Tokens/s |
| **IQ2_XS** | 52 Tokens/s | 1,110 Tokens/s |
| **Coder** | 44 Tokens/s | 1,420 Tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 mit Engine 0.1.36, die anderen Zeilen mit 0.1.26 (4K Antworten, 32K Prompts). Die vollständigen
Tabellen stehen in [DETAILS.md](docs/DETAILS.md#speed-measured). Eine Karte mit mehr VRAM ist schneller: Eine
RTX 3090 (24 GB) sollte etwa 100-140 Tokens pro Sekunde schreiben. Lange Chats und andere Karten:
[Tempo jedes Modells](docs/MODELS.md#how-fast-is-each-size), [Ergebnisse aus der Community](docs/COMMUNITY_BENCHMARKS.md).

### GLM 5.3 Flash

Gemessen für PolyStrata auf einer Workstation, mit einem Prompt von 28K Tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **3.0BIT** | 57-59 Tokens/s | 784-785 Tokens/s |
| **UD-IQ2_XXS** | 40-44 Tokens/s | 1,020 Tokens/s |
| **REAP50-Q3_K_M** | 56-62 Tokens/s | 1,250 Tokens/s |

</td></tr>
</table>

llama.cpp schrieb mit der 3.0BIT-Datei auf demselben PC 26 Tokens/s und las etwa 265. GLM 5.3 Flash ist ein viel größeres
Modell als die anderen, und nur etwa ein Fünftel seiner Experten passt in die 32 GB dieser Karte, deshalb
schreibt es langsamer. Was gemessen wurde und wie: [docs/GLM.md](docs/GLM.md#what-it-was-measured-on) (englisch).

### Qwen3.6-35B-A3B

Von PolyStrata auf zwei PCs gemessen, mit einem Prompt von 30,000 bis 32,000 Tokens: links ein Laptop mit einer
8-GB-Karte, auf dem der Prozessor die Experten rechnet, für die die Karte keinen Platz hat, rechts eine Workstation,
deren Karte das ganze Modell hält.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 73-75 Tokens/s | 1,390 Tokens/s |
| **UD-Q2_K_XL** | 91 Tokens/s | 1,320 Tokens/s |
| **UD-Q4_K_M** | 58-61 Tokens/s | 1,100-1,350 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 332-334 Tokens/s | 5,520-5,580 Tokens/s |
| **UD-Q2_K_XL** | 368-369 Tokens/s | 5,540-5,580 Tokens/s |
| **UD-Q4_K_M** | 333-337 Tokens/s | 5,630-5,660 Tokens/s |

</td></tr>
</table>

Ein kürzerer Prompt wird schneller gelesen: einer von 10,000 Tokens mit 1,450 Tokens/s auf dem Laptop und mit 5,500
auf der Workstation. llama.cpp mit dem ganzen Modell auf der Karte der Workstation schrieb etwa 180 Tokens/s und las
Prompts mit 6,000 bis 9,000. Es braucht eine Karte, auf die das Modell passt. Die Zahlen der auf 8 GB begrenzten
Workstation-Karte, ein Ersatz aus der Zeit vor der Messung auf dem Laptop, stehen in
[docs/BACKENDS.md](docs/BACKENDS.md#with-8-gb-of-vram) (englisch).

### Ornith 1.5

Genauso gemessen, auf denselben zwei PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 70 Tokens/s | 1,340 Tokens/s |
| **APEX-I-MINI** | 63-68 Tokens/s | 1,300-1,310 Tokens/s |
| **APEX-I-QUALITY** | 42-48 Tokens/s | 1,290-1,320 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 267-272 Tokens/s | 5,700-5,890 Tokens/s |
| **APEX-I-MINI** | 265-279 Tokens/s | 5,450-5,510 Tokens/s |
| **APEX-I-QUALITY** | 263-265 Tokens/s | 5,600-5,610 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,380 Tokens/s gelesen und auf der Workstation mit 5,500.
llama.cpp mit dem ganzen Modell auf der Karte der Workstation schrieb etwa 180 Tokens/s und las Prompts mit 6,000 bis
9,000.

### Gemma 4 26B-A4B

Genauso gemessen, auf denselben zwei PCs.

<table>
<tr><th>NVIDIA: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 70-71 Tokens/s | 1,910 Tokens/s |
| **UD-Q2_K_XL** | 86-87 Tokens/s | 1,840 Tokens/s |
| **UD-Q4_K_M** | 67 Tokens/s | 1,840 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 306 Tokens/s | 7,170-7,210 Tokens/s |
| **UD-Q2_K_XL** | 315 Tokens/s | 6,920 Tokens/s |
| **UD-Q4_K_M** | 301 Tokens/s | 7,120-7,210 Tokens/s |

</td></tr>
</table>

Ein Prompt von 10,000 Tokens wird auf dem Laptop mit 1,950 Tokens/s gelesen und auf der Workstation mit 5,500.
llama.cpp mit dem ganzen Modell auf der Karte der Workstation schrieb etwa 165 Tokens/s und las Prompts mit 6,000 bis
9,000.

Die vier Modelle der eigenen Engine von PolyStrata wurden mit ihrem Benchmark gemessen, den du auch auf deinem PC
laufen lassen kannst: `python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([was er schickt und ausgibt](docs/BACKENDS.md#benchmark), englisch).

## Was du brauchst

### Qwen3.8-Flash-Next

| | |
| --- | --- |
| **Grafikkarte** | **NVIDIA** GeForce RTX 20, 30, 40 oder 50 Serie, oder **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 oder RX 6800 / 6900 Serie. Sie braucht **12 GB VRAM oder mehr**. |
| **RAM** | 32 GB oder mehr. Dein RAM entscheidet, [welches Modell](#welches-modell-soll-ich-nehmen) passt. Mit 64 GB laufen alle Größen. |
| **Festplatte** | Etwa 80 GB frei. Nimm wenn möglich eine SSD: Der erste Start geht dann viel schneller. |
| **System** | Windows 10 / 11 oder Linux und ein aktueller Grafiktreiber von NVIDIA oder AMD. |

Alles andere richtet der Installer ein. Zwei oder drei Karten können sich das Modell teilen ([Multi-GPU](docs/MULTI_GPU.md)).

Experimentell, von Community-Mitgliedern auf ihren eigenen Rechnern geschrieben und getestet:

- **Ältere Grafikkarten** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Ältere GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, unter Linux aus dem Quellcode gebaut: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, unter Linux aus dem Quellcode gebaut: [Strix Halo](docs/STRIX_HALO.md).
- **Ältere Prozessoren ohne AVX2**: Sie funktionieren, aber langsam. [Ältere CPUs](docs/INSTALL.md#older-cpus-experimental).

Die vollständige Liste: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

### GLM 5.3 Flash

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer. Bisher lief es nur mit **32 GB VRAM**. Etwa 12 GB sind belegt, bevor ein Experte geladen ist, eine 12-GB-Karte kann es also nicht starten. |
| **RAM** | Neben einer 32-GB-Karte **96 GB oder mehr** für das ganze Modell, 82 GB für seine 2-Bit-Größe, 60 GB für die Größe mit der Hälfte der Experten. Eine kleinere Karte braucht mehr RAM ([wie viel](#welches-modell-soll-ich-nehmen)). Gelaufen ist es mit 120 GB. |
| **Festplatte** | 114 GB, 102 GB oder 79 GB für die Modelldatei, auf einer SSD. |
| **System** | Linux und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein. Er kompiliert die Engine auf deinem PC und bietet an, zu installieren,
was dafür fehlt (g++ und das CUDA-Toolkit). Noch nicht: Windows, AMD-Karten, mehrere Karten, Bilder.

Die vollständige Liste: [docs/GLM.md](docs/GLM.md#what-you-need) (englisch).

### Qwen3.6-35B-A3B

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 5 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. |
| **RAM** | Etwa **17, 21 oder 27 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die beiden kleineren Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 13, 17 oder 23 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows kompiliert er die Engine mit dem 2022er
Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides zu installieren. Noch nicht: AMD-Karten, mehrere
Karten.

### Ornith 1.5

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 4 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. |
| **RAM** | Etwa **18, 22 oder 28 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Eine Karte mit 24 GB hält die beiden kleineren Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 14, 17 oder 24 GB für die Modelldatei, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows kompiliert er die Engine mit dem 2022er
Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides zu installieren. Noch nicht: AMD-Karten, mehrere
Karten.

### Gemma 4 26B-A4B

| | |
| --- | --- |
| **Grafikkarte** | Eine **NVIDIA** GeForce RTX 20 oder neuer mit **8 GB VRAM oder mehr**. Etwa 6 GB sind belegt, bevor ein Experte geladen ist. Gelaufen ist es auf einer 32-GB-Karte, auf dieser Karte, auf 8 GB begrenzt, und auf der 8-GB-Karte eines Laptops. |
| **RAM** | Etwa **15, 18 oder 21 GB** neben einer 8-GB-Karte für die drei Größen, nach der Schätzung des Installers. Nach derselben Schätzung hält eine Karte mit 16 GB die beiden kleineren Größen ganz. Alle drei Größen liefen auf einem Laptop mit 32 GB. |
| **Festplatte** | 11, 13 oder 17 GB für die Modelldatei und ihr Entwurfsmodell, 1 GB mehr, um Bilder zu lesen. |
| **System** | Linux oder Windows 10 / 11 und ein aktueller NVIDIA-Treiber. |

Alles andere richtet der Installer ein, wie bei GLM 5.3 Flash. Unter Windows kompiliert er die Engine mit dem 2022er
Compiler von Visual Studio und dem CUDA-Toolkit und bietet an, beides zu installieren. Noch nicht: AMD-Karten, mehrere
Karten.

Was jedes der drei braucht und wie es geprüft wurde: [docs/BACKENDS.md](docs/BACKENDS.md#install) (englisch).

## Installieren

### Lass deine KI es einrichten

Nutzt du einen KI-Coding-Assistenten (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Füge dort diesen Text ein:

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

Er prüft deine Grafikkarte, deinen RAM und deine Festplatte und wählt das passende Modell. Dann installiert und
startet er es und sagt dir, wie du deine Apps verbindest. KI-Tools können PolyStrata auch über seinen
[MCP-Server](docs/MCP_SERVER.md) installieren, starten und stoppen.

### Oder mach es selbst

[Lade PolyStrata herunter](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) und entpacke es (oder nutze
`git clone`). **Windows:** Doppelklick auf **`START-HERE.bat`**. **Linux:** Führe im PolyStrata-Ordner **`./setup.sh`** aus.

Die Schritte sind für NVIDIA und AMD gleich. Der Installer erkennt deine Karte und richtet die passende Engine
dafür ein. Er stellt dir ein paar Fragen:

- welches Modell und welche Größe,
- wie viel Kontext (wie viel Text sich das Modell merkt),
- ob es Bilder lesen soll (GLM 5.3 Flash liest keine).

Drück jedes Mal Enter, um die empfohlene Antwort zu nehmen. Dann lädt er das Modell herunter (etwa 70 GB für
Qwen3.8-Flash-Next, 79 bis 114 GB für GLM 5.3 Flash, 11 bis 24 GB für die anderen drei) und
startet es. Bricht der Download ab, starte ihn einfach neu: Er macht dort weiter, wo er aufgehört hat. Dein Browser
öffnet die PolyStrata-App unter `http://127.0.0.1:8080`.

Alle fünf stehen in der Modellliste des Installers. `./setup.sh --family glm`, `qwen36`, `ornith` oder `gemma4` führt
direkt zu einem davon ([was er fragt und schreibt](docs/GLM.md#install), englisch).

> **Während das Modell startet, kann dein PC 1-3 Minuten langsam sein oder nicht reagieren** (beim ersten Mal am
> längsten). PolyStrata lädt 35-55 GB in deinen RAM und reserviert einen Teil davon für die Grafikkarte. Das ist normal.
> Warte und schließ das Fenster nicht. Im Fenster siehst du, was PolyStrata gerade tut.

**Beim nächsten Mal** startest du wieder `START-HERE.bat` (oder `./setup.sh`). Es startet sofort und lädt nichts
doppelt herunter. Schließ das Fenster, um das Modell zu stoppen. `UPDATE.bat` (`./update.sh`) aktualisiert PolyStrata,
ohne es zu starten. Updates, Docker, mehrere Karten, wo die Dateien landen und alle Optionen:
[docs/INSTALL.md](docs/INSTALL.md).

## Welches Modell soll ich nehmen?

### Qwen3.8-Flash-Next

Der Installer empfiehlt eins passend zu deinem RAM. Dasselbe Modell gibt es in mehreren Größen, mehr oder weniger
stark komprimiert. Kleinere Größen sind schneller. Größere sind etwas klüger.

| Dein RAM | Nimm | Warum |
| --- | --- | --- |
| **32 GB** | **Coder** | passt in 32 GB und ist für Code gemacht (mit einer 24-GB-Karte laufen auch Q2_0 und IQ2_XS) |
| **48 GB** | **IQ2_XS** (oder Q2_0, das schnellste) | die größeren Größen passen nicht |
| **64 GB** | **IQ2_XS** (empfohlen) oder IQ3_XXS / IQ3_S | alle Größen passen; IQ3_S ist das beste und das langsamste |
| **96 GB oder mehr** | **IQ3_S** oder Unsloths UD-IQ4_XS (~4 Bit) | Platz für die größten Größen, auch wenn alles andere offen ist |

- **[Coder](docs/MODELS.md#coder):** eine Version für Code, bei der die Hälfte der Experten entfernt wurde. Sie
  erreicht 91 % des SWE-bench-Verified-Werts des vollen Modells (von ihren Autoren gemessen) und passt in 32 GB RAM.
  Außerhalb von Code ist sie schwächer, auch bei chinesischem und anderem CJK-Text (#438). Nimm dafür Q2_0, IQ2_XS
  oder IQ3_S, die alle Experten behalten.
- **[Swift 1.5](docs/MODELS.md#swift-15):** ein Fine-Tune, das viel kürzer nachdenkt, bevor es antwortet. Du
  bekommst die Antwort früher, bei etwa gleicher Qualität.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** die ~4-Bit-Version von Unsloth, in der Qualität
  zwischen IQ3_S und UD-Q4_K_XL. 94 GB Download. Mit weniger als ~80 GB RAM liest PolyStrata einen Teil davon während
  der Antwort von der SSD, dort ist es also langsamer (eine NVMe-SSD hilft).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimentell): am nächsten am vollen
  Modell. Aber PolyStrata liest das meiste davon während der Antwort von der SSD, deshalb schreibt es auf einem PC mit
  64 GB nur 7-8.5 Tokens/s.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** Das richtest du von Hand
  ein. Es steht nicht im Menü des Installers.

Größen, Downloads und was wohin passt: [docs/MODELS.md](docs/MODELS.md). Um später ein weiteres Modell
hinzuzufügen, starte `SETUP.bat` (Linux: `./setup.sh --setup`).

### GLM 5.3 Flash

Bisher drei Größen. Der Installer nimmt die erste, für die dein RAM reicht.

| Größe | Was es ist | Download | RAM neben einer 32-GB-Karte |
| --- | --- | ---: | ---: |
| **3.0BIT** | das ganze Modell mit etwa 3 Bit ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | das ganze Modell mit etwa 2 Bit ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): Es liest einen Prompt schneller und schreibt langsamer | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | die Hälfte der gerouteten Experten entfernt ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) von patrickbdevaney) | 79 GB | 60 GB |

Wie viel RAM es braucht, entscheidet deine Grafikkarte: Die Experten, die nicht auf die Karte passen, müssen im RAM bleiben.

| Deine Grafikkarte | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | der PC, auf dem es gelaufen ist (120 GB RAM) |
| **24 GB** | etwa 104 GB | etwa 90 GB | etwa 68 GB | noch nicht gelaufen |
| **16 GB** | etwa 113 GB | etwa 99 GB | etwa 77 GB | noch nicht gelaufen |
| **12 GB oder weniger** | | | | startet nicht |

Die RAM-Zahlen sind die Schätzung des Installers, von dem einen PC, auf dem es gelaufen ist. Mit weniger RAM startet es
trotzdem, liest dann aber während der Antwort Experten von der SSD und wird viel langsamer. Der Installer sagt das und
fragt vorher.

Andere Quantisierungen von GLM 5.3 Flash werden noch nicht unterstützt: [was fehlt](docs/GLM.md#what-is-missing) (englisch).

### Qwen3.6-35B-A3B

Bisher drei Größen. Der Installer nimmt UD-Q3_K_M, auf einem PC mit weniger RAM UD-Q2_K_XL. UD-Q4_K_M wählt man mit Namen: `--model UD-Q4_K_M`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | das ganze Modell mit 3-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | etwa 21 GB |
| **UD-Q2_K_XL** | dasselbe mit 2- und 3-Bit-Experten: am wenigsten RAM | 13 GB | etwa 17 GB |
| **UD-Q4_K_M** | dasselbe mit 4-Bit-Experten | 23 GB | etwa 27 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit UD-Q3_K_M etwa 17 GB neben einer 12-GB-Karte, 13 GB neben einer 16-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

### Ornith 1.5

Bisher drei Größen. Der Installer nimmt APEX-I-COMPACT, auf einem PC mit weniger RAM APEX-I-MINI. APEX-I-QUALITY wählt man mit Namen: `--model APEX-I-QUALITY`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | das ganze Modell mit 3- und 4-Bit-Experten ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) von mudler) | 17 GB | etwa 22 GB |
| **APEX-I-MINI** | dasselbe mit 2- und 3-Bit-Experten: am wenigsten RAM | 14 GB | etwa 18 GB |
| **APEX-I-QUALITY** | dasselbe mit 4- bis 6-Bit-Experten | 24 GB | etwa 28 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit APEX-I-COMPACT etwa 17 GB neben einer 12-GB-Karte, 13 GB neben einer 16-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

### Gemma 4 26B-A4B

Bisher drei Größen. Der Installer nimmt UD-Q3_K_M, auf einem PC mit weniger RAM UD-Q2_K_XL. UD-Q4_K_M wählt man mit Namen: `--model UD-Q4_K_M`.

| Größe | Was es ist | Download | RAM neben einer 8-GB-Karte |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | das ganze Modell mit 3-Bit-Experten ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)) und Googles Entwurfsmodell dafür | 13 GB | etwa 18 GB |
| **UD-Q2_K_XL** | dasselbe mit 2-Bit-Experten: am wenigsten RAM | 11 GB | etwa 15 GB |
| **UD-Q4_K_M** | dasselbe mit 4- und 5-Bit-Experten | 17 GB | etwa 21 GB |

Eine größere Karte hält mehr vom Modell und braucht weniger RAM: mit UD-Q3_K_M etwa 14 GB neben einer 12-GB-Karte und 10 GB, sobald die Karte alles hält. Das sind Schätzungen des Installers. Auf einem
solchen PC wurden sie nicht geprüft.

## So benutzt du es

Alle fünf Modelle benutzt du gleich: dieselbe App, dieselbe API.

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Der Monitor-Tab der PolyStrata-App neben einem Coding-Agent"><br>
<sub>Der <b>Monitor</b> der PolyStrata-App (links), während ein Coding-Agent den Pagodengarten aus dem Video schreibt (rechts). Ein Screenshot aus <a href="https://github.com/Niko1221/Strata">Strata</a>.</sub></p>

- **Im Browser:** Öffne `http://127.0.0.1:8080`. Dort gibt es **Chat**, einen Live-**Monitor** für das Modell und
  deine GPU/CPU/RAM und **About** mit den Einstellungen und Adressen.
- **Deine Apps und Coding-Agents:** Füge einen „OpenAI-kompatiblen“ Anbieter mit der Basis-URL
  **`http://127.0.0.1:8080/v1`** hinzu. Jeder API-Key und jeder Modellname funktioniert.
  - Apps, die die API von Anthropic nutzen: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI und andere Apps, die die OpenAI Responses API nutzen: `/v1/responses`
    ([Einrichtung](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Nachdenken:** Wähle **aus, niedrig, mittel oder hoch** im Chat-Menü oder unter „Reasoning Effort“ in deiner
  App. Aus ist am schnellsten. Hoch ist am besten für schwierige Fragen.
- **Vom Handy oder einem anderen PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Setz immer einen
  Key.

Wo sie sich unterscheiden:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4 26B-A4B |
| --- | --- | --- | --- |
| **Bilder** | Antworte im Setup mit Ja auf „Images?“. Klick dann im Chat auf **Picture** oder hänge Bilder in deiner App an. AMD-Karten lesen Bilder unter Linux über den Prozessor; unter Windows geht das noch nicht. | Noch nicht. | Antworte im Setup mit Ja auf „Images?“, dann wie bei Qwen3.8-Flash-Next. Auf einer Karte mit weniger als 16 GB läuft der Bild-Encoder auf dem Prozessor, und ein Bild dauert ein paar Sekunden. |
| **Eine Anfrage nach der anderen** | Standardmäßig beantwortet PolyStrata eine Anfrage, die anderen warten. Um mehrere gleichzeitig zu beantworten, setz `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Auf einer 12-GB-Karte wird dadurch jede Antwort langsamer. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. |
| **Mehrere Unterhaltungen** | Die letzte bleibt erhalten, Folgenachrichten starten also in Sekunden. Zwei Engine-Argumente halten mehrere im RAM ([Details](docs/DETAILS.md#using-it)). | Die letzte bleibt erhalten. Das Setup fragt, ob mehrere im RAM bleiben sollen, damit ein Agent und seine Subagenten sich abwechseln, ohne dass ihre Prompts neu gelesen werden ([wie](docs/GLM.md#several-conversations-at-once)). | Wie GLM 5.3 Flash. |
| **Lange Prompts** | PolyStrata liest die erste Nachricht eines Chats komplett, etwa 1 Minute pro 30,000 Tokens. Folgenachrichten starten in Sekunden. | Die erste Nachricht wird komplett gelesen, etwa 40 Sekunden pro 30,000 Tokens. | Die erste Nachricht wird ganz gelesen: 30,000 Tokens dauern auf der Workstation 6 bis 8 Sekunden und auf dem 8-GB-Laptop 16 bis 22 Sekunden. |

Mehr: [wo deine Chats gespeichert werden](docs/INSTALL.md#where-things-are-stored), [die API](docs/DETAILS.md#using-it).

## Etwas ist schiefgelaufen?

- **Mein PC ist beim ersten Start von PolyStrata eingefroren.** Das ist normal, während das Modell lädt. Warte und
  schließ das Fenster nicht. Nach 10 Minuten immer noch eingefroren? Starte den PC neu, schließ andere Programme und
  versuch es noch einmal, oder nimm eine kleinere Größe.
- **Es hat beim Herunterladen oder Installieren aufgehört.** Starte `START-HERE.bat` (oder `./setup.sh`) noch
  einmal. Es macht dort weiter, wo es aufgehört hat.
- **Es ist sehr langsam und die Festplattenlampe blinkt ständig, oder es meldet „the engine stopped
  unexpectedly“.** Dein PC hat nicht genug freien RAM. Schließ andere Programme (Browser brauchen viel) oder nimm
  eine kleinere Größe (Q2_0 oder IQ2_XS).
- **Es meldet, dass Port 8080 schon belegt ist.** PolyStrata läuft bereits. Such nach seinem Fenster.

Weitere Probleme und ihre Lösungen: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Kommst du nicht weiter?
Öffne ein [Issue](https://github.com/VecSzn/PolyStrata/issues) und hänge `strata-<model>.log` aus dem PolyStrata-Ordner
an. Du hast ein Sicherheitsproblem gefunden? Melde es vertraulich: [SECURITY.md](SECURITY.md).

## Wie funktioniert es?

Modelle wie dieses laufen normalerweise auf Servern mit Hunderten Gigabyte Grafikspeicher. Deine Grafikkarte hat
8-32 GB. PolyStrata bringt das Modell trotzdem unter, indem es **die Arbeit auf deinen ganzen PC verteilt**. Stell dir
eine Küche vor: Was du ständig brauchst, bleibt auf der Arbeitsfläche, der Rest wartet in der Vorratskammer.

### Qwen3.8-Flash-Next

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Die 24,576 Experten des Modells: die meistgenutzten auf der Grafikkarte, alle im RAM, eine Nachschlagetabelle auf der SSD"></p>

- **Das Modell ist ein Team aus 24,576 kleinen Spezialisten („Experten“).** Jedes Wort braucht nur 10 davon.
- **Deine Grafikkarte** hält die paar tausend Experten, die am häufigsten gebraucht werden. **Dein RAM** hält alle,
  und **dein Prozessor** rechnet gleichzeitig am Rest. **Deine SSD** hält eine große Nachschlagetabelle.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Ein kleiner Helfer rät die nächsten Wörter; das große Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Ein kleiner Helfer rät die nächsten paar Wörter. Das große Modell prüft sie alle auf
  einmal. Du bekommst dieselbe Antwort, nur 1.6-1.8x schneller.
- **Lange Texte werden in großen Stücken gelesen** (bis zu 8,192 Tokens auf einmal), mit über 1,000 Tokens pro
  Sekunde.

Die längere Erklärung: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Jeder Teil und seine Zahlen:
[die Details](docs/DETAILS.md#how-it-works) und das [Paper](docs/paper/Strata-Paper.pdf).

### GLM 5.3 Flash

<p align="center"><img src="docs/media/how-it-works-glm.svg" width="860" alt="Die 12,096 Experten des Modells: die meistgenutzten auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 12,096 kleinen Experten**, 288 in jeder von 42 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht.
- **Deine Grafikkarte** hält die am häufigsten gebrauchten, etwa 2,470 auf einer 32-GB-Karte: Die Engine zählt, welche
  deine Arbeit braucht, und wählt beim nächsten Start neu. **Dein RAM** hält die übrigen, und **dein Prozessor**
  rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne zweite
  Kopie; ein Start dauert etwa 6 Sekunden, sobald die Datei in deinem RAM ist.

<p align="center"><img src="docs/media/guess-and-check-glm.svg" width="860" alt="Ein Block des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Ein Block des Modells, der zum Raten gemacht ist, schlägt die nächsten paar Wörter vor. Das
  Modell prüft sie alle auf einmal. Du bekommst dieselbe Antwort, nur 1.1-1.2x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal; die Experten, die nicht auf der Karte
  liegen, laufen durch sie), mit 780 bis 1,250 Tokens pro Sekunde.

Jeder Teil und seine Zahlen: [docs/GLM.md](docs/GLM.md#how-it-uses-the-card) (englisch).

### Qwen3.6-35B-A3B

<p align="center"><img src="docs/media/how-it-works-qwen36.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 10,240 kleinen Experten**, 256 in jeder von 40 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 2,000 auf einer 8-GB-Karte, ab 24 GB alle. Während es
  schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der 8-GB-Karte
  liefern die Experten der Karte etwa 55 % dessen, was eine Antwort braucht. **Dein RAM** hält die übrigen, und **dein
  Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne
  zweite Kopie.

<p align="center"><img src="docs/media/guess-and-check-qwen36.svg" width="860" alt="Der Entwurfsblock des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Der eigene Entwurfsblock des Modells, der in seiner Datei liegt, schlägt die nächsten paar
  Wörter vor. Das Modell prüft sie alle auf einmal. Du bekommst dieselbe Antwort, nur 1.2-1.5x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works) (englisch).

### Ornith 1.5

<p align="center"><img src="docs/media/how-it-works-ornith.svg" width="860" alt="Die 10,240 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Die Architektur von Qwen3.6-35B-A3B mit anderen Gewichten:** ein Team aus 10,240 kleinen Experten, 256 in jeder
  von 40 Schichten, und nur 8 aus jeder Schicht für jedes Wort.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 2,500 auf einer 8-GB-Karte, ab 24 GB alle. Während es
  schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und auf der 8-GB-Karte
  liefern die Experten der Karte etwa 60 % dessen, was eine Antwort braucht. **Dein RAM** hält die übrigen, und **dein
  Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei, die dort benutzt wird, wo sie liegt, ohne
  zweite Kopie.

<p align="center"><img src="docs/media/guess-and-check-ornith.svg" width="860" alt="Der Entwurfsblock des Modells rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Auch sein Entwurfsblock liegt in seiner Datei und schlägt die nächsten paar Wörter vor. Er
  rät seltener richtig als der von Qwen3.6-35B-A3B, also bekommst du dieselbe Antwort 1.1-1.2x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works) (englisch).

### Gemma 4 26B-A4B

<p align="center"><img src="docs/media/how-it-works-gemma4.svg" width="860" alt="Die 3,840 Experten des Modells: die, die eine Antwort immer wieder braucht, auf der Grafikkarte, die übrigen im RAM, die Modelldatei auf der SSD"></p>

- **Das Modell ist ein Team aus 3,840 kleinen Experten**, 128 in jeder von 30 Schichten. Jedes Wort braucht nur 8 aus
  jeder Schicht, neben einem Teil der Schicht, den jedes Wort braucht.
- **Deine Grafikkarte** hält so viele, wie passen: etwa 640 auf einer 8-GB-Karte, nach der Schätzung des Installers ab
  16 GB alle. Während es schreibt, holt die Engine die Experten auf die Karte, die die Antwort immer wieder wählt, und
  auf der 8-GB-Karte liefern die Experten der Karte etwa 55 % dessen, was eine Antwort braucht. **Dein RAM** hält die
  übrigen, und **dein Prozessor** rechnet sie gleichzeitig. **Deine SSD** hält die Modelldatei und ihr Entwurfsmodell,
  die dort benutzt werden, wo sie liegen, ohne zweite Kopie.

<p align="center"><img src="docs/media/guess-and-check-gemma4.svg" width="860" alt="Googles kleines Entwurfsmodell rät die nächsten Wörter; das Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Googles kleines Entwurfsmodell dafür, eine Datei von 0.5 GB, die der Installer mit dem
  Modell herunterlädt, schlägt die nächsten paar Wörter vor. Das Modell prüft sie alle auf einmal. Du bekommst
  dieselbe Antwort, nur 1.3-1.6x schneller.
- **Lange Texte werden in großen Stücken gelesen** (1,024 Tokens auf einmal). Hält die Karte nicht alle Experten,
  laufen die übrigen für den ganzen Prompt einmal durch sie.

Jeder Teil und seine Zahlen: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works) (englisch).

## Danksagung und Lizenz

Qwen3.8-Flash-Next ist vom [Qwen-Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Komprimiert wurde es von
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) und Unsloth.
GLM 5.3 Flash läuft aus den Dateien von [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF),
Unsloth und patrickbdevaney. Qwen3.6-35B-A3B ist vom Qwen-Team und Gemma 4 von Google, beide in der Quantisierung von
[Unsloth](https://huggingface.co/unsloth). Ornith 1.5 ist von ornith-ai, in der APEX-Quantisierung von
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF). PolyStrata baut auf [Strata](https://github.com/Niko1221/Strata) von Niko1221 und seinen Mitwirkenden
auf und nutzt Teile von [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Alle Danksagungen:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). PolyStrata ist Open Source unter der [MIT-Lizenz](LICENSE). Ein
paar Teile und alle Modelle haben eigene Lizenzen ([welche](docs/HOW_IT_WORKS.md#license)).
