<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>大きな AI モデルを、手元の PC で動かす</b><br>
Qwen3.8-Flash-Next、GLM 5.3 Flash、Qwen3.6、Ornith 1.5、Gemma 4 · 無料のオープンソース</p>

<p align="center"><img src="docs/media/speed.svg" width="820" alt="回答を書く速さ（トークン/秒）：8 GB のノート PC で Qwen3.6-35B-A3B は 73〜75、RTX 5090 で 332〜334"></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="PolyStrata のモデルが書いたボクセルの五重塔の庭。ブラウザで動いている様子"></a><br>
<sub>ボクセルの五重塔の庭。1 回のプロンプトで作成し、RTX 5070 上の <a href="https://github.com/Niko1221/Strata">Strata</a>（Qwen3.8-Flash-Next IQ3_S、128K コンテキスト）で実行 ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">動画全体（49 秒）</a></sub></p>

PolyStrata は、ふつうはサーバーが必要な大きな AI モデルを普通の PC で動かします。チャットをしたり、コードを書いたり、
アプリやコーディングエージェントと連携したりできます。データが PC の外に出ることはありません。

動かせるモデルは 5 つで、エンジンは 2 つです。

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**：12 GB 以上の NVIDIA または AMD の
  グラフィックカード、Windows または Linux。画像も読めます。エンジンは
  [Strata](https://github.com/Niko1221/Strata) のものをそのまま使っています。

残りの 4 つは PolyStrata のために書いたエンジンで動きます。NVIDIA カード 1 枚が必要です。4 つとも Linux で動き、
小さい 3 つは Windows でも動きます。まだ実験的で、動かしたのは 32 GB のカードを載せた 1 台の PC と、
小さい 3 つについては 8 GB のカードを載せたノート PC 1 台だけです。

- **[GLM 5.3 Flash](docs/GLM.md)**：いちばん大きいモデル。ファイルは 79〜114 GB で、その大部分は RAM に置かれます。
- **[Qwen3.6-35B-A3B](docs/BACKENDS.md)**：ファイルは 13〜23 GB。画像も読めます。
- **[Ornith 1.5](docs/BACKENDS.md)**：構造は Qwen3.6-35B-A3B と同じで、重みが違います。ファイルは 14〜24 GB。画像も読めます。
- **[Gemma 4 26B-A4B](docs/BACKENDS.md)**：ファイルは 11〜17 GB。画像も読めます。

サーバーは動かしたまま、インストール済みのモデルを切り替えられます（[方法](docs/BACKENDS.md#changing-the-model)、英語）。

PolyStrata は Niko1221 の [Strata](https://github.com/Niko1221/Strata) をもとにしています。

## どのくらい速い？

1 トークンは英単語のおよそ ¾ です。

- **回答の書き出し：** 短いチャットで返答が表示される速さです。毎秒 60 トークンあれば、読むより速く表示されます。
- **プロンプトの読み込み：** 送った内容を取り込む速さです（ここでは 32K トークンの文書、コード、またはチャット履歴）。

### Qwen3.8-Flash-Next

[Strata](https://github.com/Niko1221/Strata) がごく普通のゲーミング PC 2 台で測った値です。プロンプトは 32K トークン。

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA：Q2_0 はエンジン 0.1.36、ほかの行は 0.1.26 で測定（回答 4K、プロンプト 32K）。表の全体は
[DETAILS.md](docs/DETAILS.md#speed-measured) にあります。VRAM が多いカードほど速くなります。RTX 3090（24 GB）なら
毎秒およそ 100-140 トークンで書き出せるはずです。長いチャットやほかのカードについて：[モデルごとの速度](docs/MODELS.md#how-fast-is-each-size)、
[コミュニティの測定結果](docs/COMMUNITY_BENCHMARKS.md)。

### GLM 5.3 Flash

PolyStrata が 1 台のワークステーションで測った値です。プロンプトは 28K トークン。

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **3.0BIT** | 57-59 tokens/s | 784-785 tokens/s |
| **UD-IQ2_XXS** | 40-44 tokens/s | 1,020 tokens/s |
| **REAP50-Q3_K_M** | 56-62 tokens/s | 1,250 tokens/s |

</td></tr>
</table>

同じ PC で llama.cpp に 3.0BIT のファイルを読ませると、書き出しは毎秒 26 トークン、読み込みは約 265 でした。GLM 5.3 Flash は
ほかのモデルよりずっと大きく、このカードの 32 GB にはエキスパートの約 5 分の 1 しか載らないため、書き出しは遅くなります。
測定の内容と方法：[docs/GLM.md](docs/GLM.md#what-it-was-measured-on)（英語）。

### Qwen3.6-35B-A3B

PolyStrata が 2 台の PC で測りました。プロンプトは 3 万〜3.2 万トークンです。左は 8 GB のカードを載せたノート PC で、
カードに載らないエキスパートはプロセッサーが計算します。右はカードにモデル全体が載るワークステーションです。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 73-75 tokens/s | 1,390 tokens/s |
| **UD-Q2_K_XL** | 91 tokens/s | 1,320 tokens/s |
| **UD-Q4_K_M** | 58-61 tokens/s | 1,100-1,350 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 332-334 tokens/s | 5,520-5,580 tokens/s |
| **UD-Q2_K_XL** | 368-369 tokens/s | 5,540-5,580 tokens/s |
| **UD-Q4_K_M** | 333-337 tokens/s | 5,630-5,660 tokens/s |

</td></tr>
</table>

プロンプトが短いほど速く読めます。10,000 トークンのプロンプトは、ノート PC で毎秒 1,450 トークン、
ワークステーションで毎秒 5,500 トークンです。ワークステーションのカードにモデル全体を載せた llama.cpp は、
毎秒約 180 トークンで書き、プロンプトを毎秒 6,000〜9,000 トークンで読みました。llama.cpp にはモデルが収まるカードが必要です。
ワークステーションのカードを 8 GB に制限した数字は、ノート PC を測る前の代わりの値で、
[docs/BACKENDS.md](docs/BACKENDS.md#with-8-gb-of-vram)（英語）にあります。

### Ornith 1.5

同じ方法、同じ 2 台の PC で測りました。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 70 tokens/s | 1,340 tokens/s |
| **APEX-I-MINI** | 63-68 tokens/s | 1,300-1,310 tokens/s |
| **APEX-I-QUALITY** | 42-48 tokens/s | 1,290-1,320 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 267-272 tokens/s | 5,700-5,890 tokens/s |
| **APEX-I-MINI** | 265-279 tokens/s | 5,450-5,510 tokens/s |
| **APEX-I-QUALITY** | 263-265 tokens/s | 5,600-5,610 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,380 トークン、ワークステーションで毎秒 5,500 トークンで読みます。
ワークステーションのカードにモデル全体を載せた llama.cpp は、毎秒約 180 トークンで書き、
プロンプトを毎秒 6,000〜9,000 トークンで読みました。

### Gemma 4 26B-A4B

同じ方法、同じ 2 台の PC で測りました。

<table>
<tr><th>NVIDIA：RTX 4070 Laptop（8 GB）、Core i7-14700HX、RAM 32 GB</th><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB RAM</th></tr>
<tr><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 70-71 tokens/s | 1,910 tokens/s |
| **UD-Q2_K_XL** | 86-87 tokens/s | 1,840 tokens/s |
| **UD-Q4_K_M** | 67 tokens/s | 1,840 tokens/s |

</td><td>

| サイズ | 回答の書き出し | プロンプトの読み込み |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 306 tokens/s | 7,170-7,210 tokens/s |
| **UD-Q2_K_XL** | 315 tokens/s | 6,920 tokens/s |
| **UD-Q4_K_M** | 301 tokens/s | 7,120-7,210 tokens/s |

</td></tr>
</table>

10,000 トークンのプロンプトは、ノート PC で毎秒 1,950 トークン、ワークステーションで毎秒 5,500 トークンで読みます。
ワークステーションのカードにモデル全体を載せた llama.cpp は、毎秒約 165 トークンで書き、
プロンプトを毎秒 6,000〜9,000 トークンで読みました。

PolyStrata 独自のエンジンで動く 4 つのモデルは、付属のベンチマークで測りました。あなたの PC でも実行できます：
`python polystrata.py benchmark --model <name> --prompt-tokens 24000`
（[何を送り、何を表示するか](docs/BACKENDS.md#benchmark)、英語）。

## 必要なもの

### Qwen3.8-Flash-Next

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20、30、40、50 シリーズ、または **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700、RX 6800 / 6900 シリーズ。**VRAM 12 GB 以上**が必要です。 |
| **RAM** | 32 GB 以上。RAM の量で[使えるモデル](#どのモデルを選べばいい)が決まります。64 GB あればすべてのサイズが動きます。 |
| **ディスク** | 空き容量が約 80 GB。できれば SSD を使ってください。初回の起動がずっと速くなります。 |
| **OS** | Windows 10 / 11 または Linux。NVIDIA または AMD の最新のグラフィックドライバー。 |

ほかに必要なものはすべてインストーラーが用意します。2 枚や 3 枚のカードでモデルを分担することもできます（[マルチ GPU](docs/MULTI_GPU.md)）。

試験的なサポート。コミュニティのメンバーが自分のマシンで書き、テストしたものです：

- **古いグラフィックカード**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[Older GPUs](docs/OLDER_GPUS.md)。
- **Intel Arc**（Linux でソースからビルド）：[Intel Arc](docs/INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**（Linux でソースからビルド）：[Strix Halo](docs/STRIX_HALO.md)。
- **AVX2 のない古いプロセッサー**：動きますが、遅いです。[Older CPUs](docs/INSTALL.md#older-cpus-experimental)。

全リスト：[docs/INSTALL.md](docs/INSTALL.md#what-you-need)。

### GLM 5.3 Flash

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚。動かしたことがあるのは **VRAM 32 GB** だけです。エキスパートを載せる前に約 12 GB を使うので、12 GB のカードでは起動できません。 |
| **RAM** | 32 GB のカードと組み合わせる場合、モデル全体で **96 GB 以上**、2 ビットのサイズで 82 GB、エキスパートを半分にしたサイズで 60 GB。カードが小さいほど RAM が多く必要です（[どのくらいか](#どのモデルを選べばいい)）。実際に動かしたのは 120 GB です。 |
| **ディスク** | モデルファイルに 114 GB、102 GB または 79 GB。SSD に置いてください。 |
| **OS** | Linux と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。エンジンはあなたの PC でコンパイルされ、足りないもの（g++ と CUDA ツールキット）は
インストーラーが入れるか尋ねます。未対応：Windows、AMD のカード、複数のカード、画像。

全体の一覧：[docs/GLM.md](docs/GLM.md#what-you-need)（英語）。

### Qwen3.6-35B-A3B

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 5 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **17、21、27 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 13、17、23 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。Windows では Visual Studio の 2022 コンパイラーと
CUDA ツールキットでエンジンをコンパイルし、足りなければインストールするか尋ねます。未対応：AMD のカード、複数のカード。

### Ornith 1.5

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 4 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **18、22、28 GB**（インストーラーの見積もり）。24 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルに 14、17、24 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。Windows では Visual Studio の 2022 コンパイラーと
CUDA ツールキットでエンジンをコンパイルし、足りなければインストールするか尋ねます。未対応：AMD のカード、複数のカード。

### Gemma 4 26B-A4B

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20 シリーズ以降のカード 1 枚、**VRAM 8 GB 以上**。エキスパートを載せる前に約 6 GB を使います。動かしたのは 32 GB のカード、そのカードを 8 GB に制限した状態、ノート PC の 8 GB のカードです。 |
| **RAM** | 8 GB のカードと組み合わせる場合、3 つのサイズで約 **15、18、21 GB**（インストーラーの見積もり）。同じ見積もりでは、16 GB のカードなら、小さいほうの 2 サイズはモデル全体が載ります。3 つのサイズはどれも、RAM 32 GB のノート PC で動きました。 |
| **ディスク** | モデルファイルとそのドラフトモデルに 11、13、17 GB のいずれか。画像を読ませるならさらに 1 GB。 |
| **OS** | Linux または Windows 10 / 11 と、最新の NVIDIA ドライバー。 |

ほかはすべてインストーラーが用意します。GLM 5.3 Flash と同じです。Windows では Visual Studio の 2022 コンパイラーと
CUDA ツールキットでエンジンをコンパイルし、足りなければインストールするか尋ねます。未対応：AMD のカード、複数のカード。

3 つのモデルそれぞれに必要なものと確認の方法：[docs/BACKENDS.md](docs/BACKENDS.md#install)（英語）。

## インストール

### AI にセットアップしてもらう

AI コーディングアシスタント（Claude Code、Cursor、Codex、GitHub Copilot など）を使っていますか？ これを貼り付けてください：

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

AI がグラフィックカード、RAM、ディスクを調べて、合うモデルを選びます。それからインストールして起動し、
アプリのつなぎ方を教えてくれます。AI ツールは PolyStrata の [MCP サーバー](docs/MCP_SERVER.md) を通じて、
PolyStrata のインストール、起動、停止もできます。

### 自分でやる

[PolyStrata をダウンロード](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip)して展開します（または `git clone` します）。
**Windows：** **`START-HERE.bat`** をダブルクリック。**Linux：** PolyStrata フォルダーで **`./setup.sh`** を実行します。

手順は NVIDIA でも AMD でも同じです。インストーラーがカードを見つけて、それに合うエンジンを用意します。
いくつか質問があります：

- どのモデルを、どのサイズで使うか
- コンテキストをどのくらいにするか（モデルが覚えておけるテキストの量）
- 画像を読ませるかどうか（GLM 5.3 Flash は画像を読めません）

毎回 Enter を押せば、おすすめの答えになります。そのあとモデル（Qwen3.8-Flash-Next は約 70 GB、GLM 5.3 Flash は 79〜114 GB、ほかの 3 つは 11〜24 GB）をダウンロードして起動します。
ダウンロードが止まったら、もう一度実行してください。続きから再開します。ブラウザーで PolyStrata アプリが
`http://127.0.0.1:8080` に開きます。

5 つともインストーラーのモデル一覧にあります。`./setup.sh --family glm`（または `qwen36`、`ornith`、`gemma4`）なら
直接そこへ進みます（[何を尋ね、何を書き出すか](docs/GLM.md#install)、英語）。

> **モデルの起動中、1-3 分ほど PC が重くなったり、反応しなくなったりすることがあります**（初回がいちばん長いです）。
> PolyStrata は 35-55 GB を RAM に読み込み、その一部をグラフィックカード用に固定します。これは正常です。
> ウィンドウを閉じずに待ってください。PolyStrata が何をしているかはウィンドウに表示されます。

**次回からは** `START-HERE.bat`（または `./setup.sh`）をもう一度実行します。すぐに起動し、同じものを 2 回ダウンロードすることはありません。
モデルを止めるときはウィンドウを閉じます。`UPDATE.bat`（`./update.sh`）を使うと、PolyStrata を起動せずに更新できます。
更新、Docker、複数のカード、ファイルの保存場所、すべてのオプションについて：[docs/INSTALL.md](docs/INSTALL.md)。

## どのモデルを選べばいい

### Qwen3.8-Flash-Next

インストーラーが RAM に合わせてひとつ勧めます。同じモデルにも、圧縮の強さが違う複数のサイズがあります。
小さいサイズほど速く、大きいサイズほど少し賢くなります。

| RAM | 選ぶもの | 理由 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB に収まり、コード向けに作られている（24 GB のカードなら Q2_0 と IQ2_XS も動く） |
| **48 GB** | **IQ2_XS**（または最速の Q2_0） | 大きいサイズは収まらない |
| **64 GB** | **IQ2_XS**（おすすめ）、または IQ3_XXS / IQ3_S | すべてのサイズが収まる。IQ3_S がいちばん賢く、いちばん遅い |
| **96 GB 以上** | **IQ3_S**、または Unsloth の UD-IQ4_XS（約 4-bit） | ほかのものを全部開いたままでも、最大のサイズが入る余裕がある |

- **[Coder](docs/MODELS.md#coder)：** エキスパートを半分取り除いたコーディング向け版です。フルモデルの SWE-bench Verified スコアの
  91% に届き（作者による測定）、32 GB の RAM に収まります。コード以外では弱く、中国語などの CJK のテキストも苦手です（#438）。
  そうした用途には、すべてのエキスパートを残している Q2_0、IQ2_XS、IQ3_S を選んでください。
- **[Swift 1.5](docs/MODELS.md#swift-15)：** 答える前に考える時間がずっと短いファインチューン版です。
  ほぼ同じ品質のまま、答えが早く返ってきます。
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)：** Unsloth の約 4-bit 版で、品質は IQ3_S と UD-Q4_K_XL の間です。
  ダウンロードは 94 GB。RAM が約 80 GB 未満だと、答えている間に PolyStrata がその一部を SSD から読むので遅くなります（NVMe SSD が有効です）。
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)**（試験的）：フルモデルにいちばん近いものです。
  ただし答えている間、PolyStrata はその大部分を SSD から読むので、64 GB の PC では毎秒 7-8.5 トークンしか書き出せません。
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)：** 手動でセットアップします。
  インストーラーのメニューにはありません。

サイズ、ダウンロード、どれがどこに収まるかについて：[docs/MODELS.md](docs/MODELS.md)。あとで別のモデルを追加するには
`SETUP.bat`（Linux：`./setup.sh --setup`）を実行します。

### GLM 5.3 Flash

いまのところ 3 サイズです。インストーラーは、RAM が足りる最初のサイズを選びます。

| サイズ | 内容 | ダウンロード | 32 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **3.0BIT** | モデル全体、約 3 ビット（[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)） | 114 GB | 96 GB |
| **UD-IQ2_XXS** | モデル全体、約 2 ビット（[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)）。プロンプトの読み込みは速く、書き出しは遅くなります | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | ルーティングされるエキスパートを半分に減らしたもの（patrickbdevaney の [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)） | 79 GB | 60 GB |

必要な RAM はグラフィックカードで決まります。カードに載らないエキスパートは RAM に置いておく必要があるからです。

| グラフィックカード | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | 実際に動かした PC（RAM 120 GB） |
| **24 GB** | 約 104 GB | 約 90 GB | 約 68 GB | 未実行 |
| **16 GB** | 約 113 GB | 約 99 GB | 約 77 GB | 未実行 |
| **12 GB 以下** | | | | 起動しません |

RAM の数字は、動かしたことのある 1 台の PC をもとにしたインストーラーの見積もりです。RAM が足りなくても起動はしますが、
回答中にエキスパートを SSD から読むことになり、ずっと遅くなります。インストーラーはそのことを伝え、先に確認します。

GLM 5.3 Flash のほかの量子化にはまだ対応していません：[足りないもの](docs/GLM.md#what-is-missing)（英語）。

### Qwen3.6-35B-A3B

いまのところ 3 サイズです。インストーラーは UD-Q3_K_M を選び、RAM が足りない PC では UD-Q2_K_XL を選びます。UD-Q4_K_M は名前で指定します（`--model UD-Q4_K_M`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | モデル全体、エキスパートは 3 ビット（[Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)） | 17 GB | 約 21 GB |
| **UD-Q2_K_XL** | 同じモデル、エキスパートは 2 ビットと 3 ビット。必要な RAM が最も少ない | 13 GB | 約 17 GB |
| **UD-Q4_K_M** | 同じモデル、エキスパートは 4 ビット | 23 GB | 約 27 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。UD-Q3_K_M を 12 GB のカードと組む場合は約 17 GB、16 GB のカードなら 13 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

### Ornith 1.5

いまのところ 3 サイズです。インストーラーは APEX-I-COMPACT を選び、RAM が足りない PC では APEX-I-MINI を選びます。APEX-I-QUALITY は名前で指定します（`--model APEX-I-QUALITY`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | モデル全体、エキスパートは 3 ビットと 4 ビット（mudler の [APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)） | 17 GB | 約 22 GB |
| **APEX-I-MINI** | 同じモデル、エキスパートは 2 ビットと 3 ビット。必要な RAM が最も少ない | 14 GB | 約 18 GB |
| **APEX-I-QUALITY** | 同じモデル、エキスパートは 4〜6 ビット | 24 GB | 約 28 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。APEX-I-COMPACT を 12 GB のカードと組む場合は約 17 GB、16 GB のカードなら 13 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

### Gemma 4 26B-A4B

いまのところ 3 サイズです。インストーラーは UD-Q3_K_M を選び、RAM が足りない PC では UD-Q2_K_XL を選びます。UD-Q4_K_M は名前で指定します（`--model UD-Q4_K_M`）。

| サイズ | 内容 | ダウンロード | 8 GB のカードと組む場合の RAM |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | モデル全体、エキスパートは 3 ビット（[Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)）。Google がこのモデル用に出しているドラフトモデルも付きます | 13 GB | 約 18 GB |
| **UD-Q2_K_XL** | 同じモデル、エキスパートは 2 ビット。必要な RAM が最も少ない | 11 GB | 約 15 GB |
| **UD-Q4_K_M** | 同じモデル、エキスパートは 4 ビットと 5 ビット | 17 GB | 約 21 GB |

カードが大きいほどモデルを多く載せられ、必要な RAM は減ります。UD-Q3_K_M を 12 GB のカードと組む場合は約 14 GB、カードにすべて載れば 10 GB です。
これらはインストーラーの見積もりで、そのような PC では確かめていません。

## 使い方

5 つのモデルの使い方は同じです。同じアプリ、同じ API です。

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="PolyStrata アプリの Monitor タブと、その横で動くコーディングエージェント"><br>
<sub>PolyStrata アプリの <b>Monitor</b>（左）。コーディングエージェントが動画の五重塔の庭を書いているところ（右）。<a href="https://github.com/Niko1221/Strata">Strata</a> のスクリーンショットです。</sub></p>

- **ブラウザーで：** `http://127.0.0.1:8080` を開きます。**Chat**、モデルと GPU/CPU/RAM の様子をリアルタイムで見る **Monitor**、
  設定とアドレスが載った **About** があります。
- **アプリやコーディングエージェントから：** 「OpenAI-compatible」プロバイダーを追加し、ベース URL に
  **`http://127.0.0.1:8080/v1`** を指定します。API キーとモデル名は何でもかまいません。
  - Anthropic の API を使うアプリ：`http://127.0.0.1:8080/v1/messages`（Claude Code：
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）。
  - Codex CLI など OpenAI Responses API を使うアプリ：`/v1/responses`
    （[設定方法](docs/DETAILS.md#the-responses-api-and-codex-cli)）。
- **思考（Thinking）：** チャットのメニュー、またはアプリの「reasoning effort」で **off、low、medium、high** を選びます。
  off がいちばん速く、high は難しい質問に向いています。
- **スマートフォンや別の PC から：** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`。キーは必ず設定してください。

モデルによって違うところ：

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | Qwen3.6-35B-A3B、Ornith 1.5、Gemma 4 26B-A4B |
| --- | --- | --- | --- |
| **画像** | セットアップで「Images?」に yes と答えます。そのあとチャットで **Picture** をクリックするか、アプリで画像を添付します。AMD のカードは、Linux ではプロセッサーを使って画像を読みます。Windows ではまだ読めません。 | まだ読めません。 | セットアップで「Images?」に yes と答えます。あとは Qwen3.8-Flash-Next と同じです。16 GB 未満のカードでは画像エンコーダーがプロセッサーで動き、画像 1 枚に数秒かかります。 |
| **リクエストは 1 つずつ** | 初期設定では PolyStrata は 1 つのリクエストに答え、ほかは待ちます。複数に同時に答えるには`"parallel": 2` を設定します（[BATCHING.md](docs/BATCHING.md)）。12 GB のカードでは、それぞれの回答が遅くなります。 | 1 つずつ答え、ほかは待ちます。 | 1 つずつ答え、ほかは待ちます。 |
| **複数の会話** | 直前の会話は保持されるので、2 通目以降は数秒で始まります。エンジンの引数を 2 つ足すと、複数の会話を RAM に保持できます（[詳細](docs/DETAILS.md#using-it)）。 | 直前の会話は保持されます。セットアップで、複数の会話を RAM に保持するかどうかを尋ねます。保持すると、エージェントとサブエージェントが交互に使っても、プロンプトを読み直しません（[仕組み](docs/GLM.md#several-conversations-at-once)）。 | GLM 5.3 Flash と同じです。 |
| **長いプロンプト** | PolyStrata はチャットの最初のメッセージをすべて読みます。30,000 トークンあたり約 1 分かかります。2 通目以降のメッセージは数秒で始まります。 | 最初のメッセージはすべて読みます。30,000 トークンあたり約 40 秒かかります。 | 最初のメッセージは全部読みます。30,000 トークンで、ワークステーションは 6〜8 秒、8 GB のノート PC は 16〜22 秒です。 |

詳しくは：[チャットの保存場所](docs/INSTALL.md#where-things-are-stored)、[API](docs/DETAILS.md#using-it)。

## うまくいかないとき

- **PolyStrata の初回起動で PC が固まった。** モデルの読み込み中はよくあることです。ウィンドウを閉じずに待ってください。
  10 分たっても固まったまま？ PC を再起動し、ほかのプログラムを閉じてもう一度試すか、小さいサイズを選んでください。
- **ダウンロードやインストールの途中で止まった。** `START-HERE.bat`（または `./setup.sh`）をもう一度実行してください。
  止まったところから続きます。
- **とても遅く、ディスクのランプが点滅し続ける。または「the engine stopped unexpectedly」と表示される。** PC の空き RAM が
  足りません。ほかのプログラムを閉じるか（ブラウザーは多く使います）、小さいサイズ（Q2_0 または IQ2_XS）を選んでください。
- **ポート 8080 がすでに使われていると表示される。** PolyStrata はもう動いています。そのウィンドウを探してください。

ほかの問題と解決方法：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。それでも解決しない場合は
[issue](https://github.com/VecSzn/PolyStrata/issues) を作成し、PolyStrata フォルダーにある `strata-<model>.log` を添付してください。
セキュリティの問題を見つけたら、非公開で報告してください：[SECURITY.md](SECURITY.md)。

## しくみ

このようなモデルは、ふつう数百ギガバイトのグラフィックメモリを持つサーバーで動きます。あなたのグラフィックカードは
8-32 GB です。PolyStrata は **PC 全体で作業を分担する** ことで、モデルを収めています。台所を思い浮かべてください。
いつも使う道具は調理台に置き、残りは食料庫にしまっておきます。

### Qwen3.8-Flash-Next

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="モデルの 24,576 個のエキスパート：よく使うものはグラフィックカードに、すべては RAM に、ルックアップテーブルは SSD に"></p>

- **モデルは 24,576 個の小さな専門家（「エキスパート」）のチームです。** 1 語ごとに必要なのはそのうち 10 個だけです。
- **グラフィックカード** は、よく使われる数千個のエキスパートを持ちます。**RAM** はすべてのエキスパートを持ち、
  **プロセッサー** が残りを同時に処理します。**SSD** は大きなルックアップテーブルを持ちます。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="小さなヘルパーが次の単語を予想し、大きなモデルがまとめて確認して正しいものを残す"></p>

- **予想してから確認：** 小さなヘルパーが次の数語を予想します。大きなモデルがそれをまとめて確認します。
  答えは同じまま、1.6-1.8x 早く返ってきます。
- **長いテキストは大きなまとまりで読みます**（一度に最大 8,192 トークン）。速さは毎秒 1,000 トークン以上です。

もっと詳しい説明：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。各部分とその数値：
[詳細](docs/DETAILS.md#how-it-works)と[論文](docs/paper/Strata-Paper.pdf)。

### GLM 5.3 Flash

<p align="center"><img src="docs/media/how-it-works-glm.svg" width="860" alt="モデルの 12,096 個のエキスパート：よく使うものはグラフィックカードに、残りは RAM に、モデルファイルは SSD に"></p>

- **モデルは 12,096 個の小さなエキスパートのチームです。** 42 層それぞれに 288 個あります。1 単語ごとに必要なのは、各層から 8 個だけです。
- **グラフィックカード** には、よく使われるものを置きます。32 GB のカードで約 2,470 個です。エンジンはどれが使われたかを数え、
  次の起動のときに選び直します。**RAM** は残りを持ち、**プロセッサー** がそれを同時に計算します。**SSD** はモデルファイルを持ちます。
  ファイルは置いた場所でそのまま使い、2 つ目のコピーは作りません。ファイルが RAM に載っていれば、起動は約 6 秒です。

<p align="center"><img src="docs/media/guess-and-check-glm.svg" width="860" alt="モデルのブロックが次の単語を予想し、モデルがまとめて確認して正しいものを残す"></p>

- **予想してから確認：** モデルには予想のためのブロックがあり、それが次の数語を出します。モデルはそれをまとめて確認します。
  答えは同じまま、1.1-1.2x 早く返ってきます。
- **長いテキストは大きなまとまりで読みます**（一度に 1,024 トークン。カードに載っていないエキスパートはカードを通します）。
  速さは毎秒 780〜1,250 トークンです。

各部分とその数字：[docs/GLM.md](docs/GLM.md#how-it-uses-the-card)（英語）。

### Qwen3.6-35B-A3B

<p align="center"><img src="docs/media/how-it-works-qwen36.svg" width="860" alt="モデルの 10,240 個のエキスパート：回答が繰り返し使うものはグラフィックカードに、残りは RAM に、モデルファイルは SSD に"></p>

- **モデルは 10,240 個の小さなエキスパートのチームです。** 40 層それぞれに 256 個あります。1 単語ごとに必要なのは、各層から 8 個だけです。
- **グラフィックカード** には載るだけ載せます。8 GB のカードで約 2,000 個、24 GB あればすべて載ります。書き出しのあいだ、
  エンジンは回答が繰り返し選ぶエキスパートをカードへ運び、8 GB のカードでは回答が使うエキスパートの約 55% をカードがまかないます。
  **RAM** は残りを持ち、**プロセッサー** がそれを同時に計算します。**SSD** はモデルファイルを持ちます。
  ファイルは置いた場所でそのまま使い、2 つ目のコピーは作りません。

<p align="center"><img src="docs/media/guess-and-check-qwen36.svg" width="860" alt="モデルのドラフトブロックが次の単語を予想し、モデルがまとめて確認して正しいものを残す"></p>

- **予想してから確認：** モデル自身のドラフトブロックがファイルに入っていて、それが次の数語を出します。モデルはそれをまとめて確認します。
  答えは同じまま、1.2-1.5x 早く返ってきます。
- **長いテキストは大きなまとまりで読みます**（一度に 1,024 トークン）。カードにすべてのエキスパートが載らないときは、
  残りのエキスパートがプロンプト全体で 1 回だけカードを通ります。

各部分とその数字：[docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works)（英語）。

### Ornith 1.5

<p align="center"><img src="docs/media/how-it-works-ornith.svg" width="860" alt="モデルの 10,240 個のエキスパート：回答が繰り返し使うものはグラフィックカードに、残りは RAM に、モデルファイルは SSD に"></p>

- **構造は Qwen3.6-35B-A3B と同じで、重みが違います。** 10,240 個の小さなエキスパートのチームで、40 層それぞれに 256 個あり、
  1 単語ごとに各層から 8 個だけを使います。
- **グラフィックカード** には載るだけ載せます。8 GB のカードで約 2,500 個、24 GB あればすべて載ります。書き出しのあいだ、
  エンジンは回答が繰り返し選ぶエキスパートをカードへ運び、8 GB のカードでは回答が使うエキスパートの約 60% をカードがまかないます。
  **RAM** は残りを持ち、**プロセッサー** がそれを同時に計算します。**SSD** はモデルファイルを持ちます。
  ファイルは置いた場所でそのまま使い、2 つ目のコピーは作りません。

<p align="center"><img src="docs/media/guess-and-check-ornith.svg" width="860" alt="モデルのドラフトブロックが次の単語を予想し、モデルがまとめて確認して正しいものを残す"></p>

- **予想してから確認：** ドラフトブロックはこちらもファイルに入っていて、次の数語を出します。予想が当たる割合は
  Qwen3.6-35B-A3B のものより低く、同じ答えが 1.1-1.2x 早く返ってきます。
- **長いテキストは大きなまとまりで読みます**（一度に 1,024 トークン）。カードにすべてのエキスパートが載らないときは、
  残りのエキスパートがプロンプト全体で 1 回だけカードを通ります。

各部分とその数字：[docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works)（英語）。

### Gemma 4 26B-A4B

<p align="center"><img src="docs/media/how-it-works-gemma4.svg" width="860" alt="モデルの 3,840 個のエキスパート：回答が繰り返し使うものはグラフィックカードに、残りは RAM に、モデルファイルは SSD に"></p>

- **モデルは 3,840 個の小さなエキスパートのチームです。** 30 層それぞれに 128 個あります。1 単語ごとに必要なのは各層から 8 個だけで、
  そのほかにすべての単語が使う部分が各層にあります。
- **グラフィックカード** には載るだけ載せます。8 GB のカードで約 640 個、インストーラーの見積もりでは 16 GB あればすべて載ります。
  書き出しのあいだ、エンジンは回答が繰り返し選ぶエキスパートをカードへ運び、8 GB のカードでは回答が使うエキスパートの約 55% を
  カードがまかないます。**RAM** は残りを持ち、**プロセッサー** がそれを同時に計算します。**SSD** はモデルファイルとドラフトモデルを
  持ちます。どちらも置いた場所でそのまま使い、2 つ目のコピーは作りません。

<p align="center"><img src="docs/media/guess-and-check-gemma4.svg" width="860" alt="Google の小さなドラフトモデルが次の単語を予想し、モデルがまとめて確認して正しいものを残す"></p>

- **予想してから確認：** Google がこのモデル用に出している小さなドラフトモデル（0.5 GB のファイル）を、インストーラーがモデルと一緒に
  ダウンロードします。それが次の数語を出し、モデルがまとめて確認します。答えは同じまま、1.3-1.6x 早く返ってきます。
- **長いテキストは大きなまとまりで読みます**（一度に 1,024 トークン）。カードにすべてのエキスパートが載らないときは、
  残りのエキスパートがプロンプト全体で 1 回だけカードを通ります。

各部分とその数字：[docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works)（英語）。

## クレジットとライセンス

Qwen3.8-Flash-Next は [Qwen チーム](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) のモデルです。圧縮は
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）、Unsloth によるものです。
GLM 5.3 Flash は [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)、Unsloth、
patrickbdevaney のファイルで動きます。Qwen3.6-35B-A3B は Qwen チーム、Gemma 4 は Google のモデルで、どちらも
[Unsloth](https://huggingface.co/unsloth) の量子化を使っています。Ornith 1.5 は ornith-ai のモデルで、
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) の APEX 量子化を使っています。PolyStrata は Niko1221 とコントリビューターによる [Strata](https://github.com/Niko1221/Strata) をもとにしており、
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) の一部を使っています。すべてのクレジット：
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)。PolyStrata は [MIT License](LICENSE) のオープンソースです。一部のパーツと
すべてのモデルには、それぞれ独自のライセンスがあります（[どれか](docs/HOW_IT_WORKS.md#license)）。
