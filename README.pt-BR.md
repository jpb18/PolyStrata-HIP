<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

<p align="center"><b>Rode modelos de IA grandes no seu próprio PC</b><br>
Qwen3.8-Flash-Next, GLM 5.3 Flash, Qwen3.6, Ornith 1.5 e Gemma 4 · gratuito e de código aberto</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Um jardim com pagode em voxels que o modelo do PolyStrata escreveu, rodando no navegador"></a><br>
<sub>Um jardim com pagode em voxels, feito com um único prompt numa RTX 5070 com o <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

O PolyStrata roda modelos de IA grandes num PC comum, modelos que normalmente precisam de um servidor.
Eles conversam, escrevem código e funcionam com seus apps e agentes de programação. Nada sai do seu PC.

Ele roda cinco modelos, com dois motores:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** numa placa de vídeo NVIDIA ou AMD de
  12 GB ou mais, no Windows ou no Linux. Ele também entende imagens. O motor dele é o do
  [Strata](https://github.com/Niko1221/Strata), sem mudanças.

Os outros quatro usam um motor escrito para o PolyStrata, no Linux com uma placa NVIDIA. São experimentais: só
rodaram em um PC, com uma placa de 32 GB.

- **[GLM 5.3 Flash](docs/GLM.md)**, o maior: um arquivo de 79 a 114 GB, que fica na maior parte na RAM.
- **[Qwen3.6-35B-A3B](docs/BACKENDS.md)**, um arquivo de 17 GB. Ele também entende imagens.
- **[Ornith 1.5](docs/BACKENDS.md)**, com a arquitetura do Qwen3.6-35B-A3B e outros pesos: um arquivo de 17 GB. Ele
  também entende imagens.
- **[Gemma 4 26B-A4B](docs/BACKENDS.md)**, um arquivo de 13 GB. Ele também entende imagens.

O servidor pode trocar entre os modelos instalados enquanto está rodando
([como](docs/BACKENDS.md#changing-the-model), em inglês).

O PolyStrata é feito sobre o [Strata](https://github.com/Niko1221/Strata) de Niko1221.

## Qual é a velocidade?

Um token é mais ou menos ¾ de uma palavra.

- **Escreve respostas:** a velocidade com que a resposta aparece numa conversa curta. 60 tokens por segundo é mais rápido do que você consegue ler.
- **Lê seu prompt:** a velocidade com que ele absorve o que você envia (aqui, um documento, código ou histórico de conversa de 32K tokens).

### Qwen3.8-Flash-Next

Medido pelo [Strata](https://github.com/Niko1221/Strata) em dois PCs gamer comuns, com um prompt de 32K tokens.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB de RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 com o engine 0.1.36, as outras linhas com 0.1.26 (respostas de 4K, prompts de 32K). As tabelas
completas estão em [DETAILS.md](docs/DETAILS.md#speed-measured). Uma placa com mais VRAM é mais rápida: uma
RTX 3090 (24 GB) deve escrever cerca de 100-140 tokens por segundo. Conversas longas e outras placas:
[velocidade de cada modelo](docs/MODELS.md#how-fast-is-each-size), [resultados da comunidade](docs/COMMUNITY_BENCHMARKS.md).

### GLM 5.3 Flash

Medido para o PolyStrata numa estação de trabalho, com um prompt de 28K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **3.0BIT** | 58-59 tokens/s | 529-531 tokens/s |
| **UD-IQ2_XXS** (build anterior) | 33 tokens/s | 347 tokens/s |
| **REAP50-Q3_K_M** (build anterior) | 27 tokens/s | 384 tokens/s |

</td></tr>
</table>

As linhas UD-IQ2_XXS e REAP50-Q3_K_M são de builds anteriores do motor e não foram medidas de novo: os arquivos
delas não estão mais no PC de teste. Com essas duas builds, o 3.0BIT escrevia 37 e 27 tokens/s e lia 286 e 312.
No mesmo PC, o llama.cpp escreveu 26 tokens/s com o arquivo 3.0BIT e leu cerca de 265. O GLM 5.3 Flash é um modelo
muito maior que os outros, e só cerca de um quinto dos experts dele cabe nos 32 GB desta placa, por isso
ele escreve mais devagar. O que foi medido e como: [docs/GLM.md](docs/GLM.md#what-it-was-measured-on) (em inglês).

### Qwen3.6-35B-A3B

Medido para o PolyStrata na mesma estação de trabalho, com um prompt de 30K tokens. À esquerda, a placa guarda o
modelo inteiro. À direita, a mesma placa está limitada a 8 GB, e a maioria dos experts é calculada pelo processador.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th><th>O mesmo PC, com a placa limitada a 8 GB</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 310-311 tokens/s | 3,830-3,860 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 156-165 tokens/s | 3,160-3,190 tokens/s |

</td></tr>
</table>

A tabela da direita não é um PC com placa de 8 GB: o processador, a RAM e a velocidade da placa são os desta estação
de trabalho. Um prompt mais curto é lido mais rápido: um de 10,000 tokens a 4,800 tokens/s com o modelo inteiro na
placa. O llama.cpp, com o modelo inteiro na mesma placa, escreveu cerca de 180 tokens/s e leu prompts a 6,000 a 9,000.
Ele precisa de uma placa em que o modelo caiba.

### Ornith 1.5

Medido do mesmo jeito, com o mesmo prompt de 30K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th><th>O mesmo PC, com a placa limitada a 8 GB</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 229 tokens/s | 3,860 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **APEX-I-COMPACT** | 140-148 tokens/s | 3,060-3,070 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 4,800 tokens/s com o modelo inteiro na placa. O llama.cpp, com o modelo inteiro
na mesma placa, escreveu cerca de 180 tokens/s e leu prompts a 6,000 a 9,000.

### Gemma 4 26B-A4B

Medido do mesmo jeito, com um prompt de 31K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th><th>O mesmo PC, com a placa limitada a 8 GB</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 274-276 tokens/s | 3,850 tokens/s |

</td><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **UD-Q3_K_M** | 135-142 tokens/s | 3,390-3,420 tokens/s |

</td></tr>
</table>

Um prompt de 10,000 tokens é lido a 4,600 tokens/s com o modelo inteiro na placa. O llama.cpp, com o modelo
inteiro na mesma placa, escreveu cerca de 165 tokens/s e leu prompts a 6,000 a 9,000.

Os quatro modelos do motor próprio do PolyStrata foram medidos com o benchmark dele, que você pode rodar no seu PC:
`python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([o que ele envia e mostra](docs/BACKENDS.md#benchmark), em inglês).

## Do que você precisa

### Qwen3.8-Flash-Next

| | |
| --- | --- |
| **Placa de vídeo** | **NVIDIA** GeForce RTX séries 20, 30, 40 ou 50, ou **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 ou séries RX 6800 / 6900. Ela precisa de **12 GB de VRAM ou mais**. |
| **RAM** | 32 GB ou mais. Sua RAM decide [qual modelo](#qual-modelo-devo-escolher) cabe. Com 64 GB, roda qualquer tamanho. |
| **Disco** | Cerca de 80 GB livres. Use um SSD se puder: a primeira inicialização fica bem mais rápida. |
| **Sistema** | Windows 10 / 11 ou Linux, e um driver de vídeo atualizado da NVIDIA ou da AMD. |

O instalador cuida de todo o resto. Duas ou três placas podem dividir o modelo ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, escrito e testado por membros da comunidade nas próprias máquinas:

- **Placas de vídeo mais antigas** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, compilado a partir do código-fonte no Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Processadores mais antigos sem AVX2**: funcionam, mas devagar. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

A lista completa: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

### GLM 5.3 Flash

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova. Ele só rodou com **32 GB de VRAM**. Cerca de 12 GB são ocupados antes do primeiro expert, então uma placa de 12 GB não consegue iniciá-lo. |
| **RAM** | Ao lado de uma placa de 32 GB, **96 GB ou mais** para o modelo inteiro, 82 GB para o tamanho de 2 bits, 60 GB para o tamanho com metade dos experts. Uma placa menor precisa de mais RAM ([quanto](#qual-modelo-devo-escolher)). Ele rodou com 120 GB. |
| **Disco** | 114 GB, 102 GB ou 79 GB para o arquivo do modelo, num SSD. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto. Ele compila o motor no seu PC e oferece instalar o que falta para isso
(g++ e o CUDA Toolkit). Ainda não: Windows, placas AMD, várias placas, imagens.

A lista completa: [docs/GLM.md](docs/GLM.md#what-you-need) (em inglês).

### Qwen3.6-35B-A3B

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 5 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, e nessa placa limitada a 8 GB. |
| **RAM** | Cerca de **21 GB** ao lado de uma placa de 8 GB, pela estimativa do instalador. Uma placa de 24 GB guarda o modelo inteiro. |
| **Disco** | 17 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. Ainda não: Windows, placas AMD, várias placas.

### Ornith 1.5

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 4 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, e nessa placa limitada a 8 GB. |
| **RAM** | Cerca de **22 GB** ao lado de uma placa de 8 GB, pela estimativa do instalador. Uma placa de 24 GB guarda o modelo inteiro. |
| **Disco** | 17 GB para o arquivo do modelo, 1 GB a mais para ler imagens. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. Ainda não: Windows, placas AMD, várias placas.

### Gemma 4 26B-A4B

| | |
| --- | --- |
| **Placa de vídeo** | Uma placa **NVIDIA** GeForce RTX série 20 ou mais nova com **8 GB de VRAM ou mais**. Cerca de 6 GB são ocupados antes do primeiro expert. Ele rodou numa placa de 32 GB, e nessa placa limitada a 8 GB. |
| **RAM** | Cerca de **18 GB** ao lado de uma placa de 8 GB, pela estimativa do instalador. Pela mesma estimativa, uma placa de 16 GB guarda o modelo inteiro. |
| **Disco** | 13 GB para o arquivo do modelo e o modelo de rascunho dele, 1 GB a mais para ler imagens. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto, como faz com o GLM 5.3 Flash. Ainda não: Windows, placas AMD, várias placas.

Do que cada um dos três precisa e como foi conferido: [docs/BACKENDS.md](docs/BACKENDS.md#install) (em inglês).

## Instalação

### Deixe sua IA instalar

Você usa um assistente de programação com IA (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Cole isto nele:

```text
Set up PolyStrata on this PC for me: https://github.com/VecSzn/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

Ele verifica sua placa de vídeo, RAM e disco e escolhe o modelo que cabe. Depois instala, inicia e explica como
conectar seus apps. Ferramentas de IA também podem instalar, iniciar e parar o PolyStrata pelo
[servidor MCP](docs/MCP_SERVER.md) dele.

### Ou faça você mesmo

[Baixe o PolyStrata](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) e descompacte (ou use `git clone`).
**Windows:** clique duas vezes em **`START-HERE.bat`**. **Linux:** rode **`./setup.sh`** na pasta do PolyStrata.

Os passos são os mesmos para NVIDIA e AMD. O instalador encontra sua placa e configura o engine certo para ela.
Ele faz algumas perguntas:

- qual modelo e qual tamanho,
- quanto contexto (quanto texto o modelo guarda na memória),
- se ele deve entender imagens (o GLM 5.3 Flash não entende).

Aperte Enter em cada uma para usar a resposta recomendada. Depois ele baixa o modelo (cerca de 70 GB para o
Qwen3.8-Flash-Next, 79 a 114 GB para o GLM 5.3 Flash, 13 a 17 GB para os outros três) e o inicia.
Se o download parar, rode de novo: ele continua de onde parou. Seu navegador abre o app do PolyStrata em
`http://127.0.0.1:8080`.

Os cinco estão na lista de modelos do instalador. `./setup.sh --family glm`, `qwen36`, `ornith` ou `gemma4` vai
direto para um deles ([o que ele pergunta e grava](docs/GLM.md#install), em inglês).

> **Enquanto o modelo inicia, seu PC pode ficar lento ou parar de responder por 1-3 minutos** (mais na primeira vez).
> O PolyStrata carrega 35-55 GB na sua RAM e reserva parte dela para a placa de vídeo. Isso é normal. Espere e não
> feche a janela. A janela mostra o que o PolyStrata está fazendo.

**Da próxima vez**, rode `START-HERE.bat` (ou `./setup.sh`) de novo. Ele inicia na hora e não baixa nada duas
vezes. Feche a janela para parar o modelo. `UPDATE.bat` (`./update.sh`) atualiza o PolyStrata sem iniciá-lo.
Atualização, Docker, várias placas, onde ficam os arquivos e todas as opções: [docs/INSTALL.md](docs/INSTALL.md).

## Qual modelo devo escolher?

### Qwen3.8-Flash-Next

O instalador recomenda um de acordo com sua RAM. O mesmo modelo vem em vários tamanhos, mais ou menos comprimidos.
Os tamanhos menores são mais rápidos. Os maiores são um pouco mais inteligentes.

| Sua RAM | Escolha | Por quê |
| --- | --- | --- |
| **32 GB** | **Coder** | cabe em 32 GB e é feito para código (com uma placa de 24 GB, Q2_0 e IQ2_XS também rodam) |
| **48 GB** | **IQ2_XS** (ou Q2_0, o mais rápido) | os tamanhos maiores não cabem |
| **64 GB** | **IQ2_XS** (recomendado), ou IQ3_XXS / IQ3_S | todos os tamanhos cabem; IQ3_S é o melhor e o mais lento |
| **96 GB ou mais** | **IQ3_S**, ou o UD-IQ4_XS da Unsloth (~4-bit) | espaço para os maiores tamanhos com todo o resto aberto |

- **[Coder](docs/MODELS.md#coder):** uma versão para programação com metade dos experts removida. Ela alcança 91%
  da pontuação do modelo completo no SWE-bench Verified (medido pelos autores) e cabe em 32 GB de RAM. É mais fraca
  fora do código, inclusive em chinês e outros textos CJK (#438). Para isso, escolha Q2_0, IQ2_XS ou IQ3_S, que
  mantêm todos os experts.
- **[Swift 1.5](docs/MODELS.md#swift-15):** um fine-tune que pensa por bem menos tempo antes de responder. Você
  recebe a resposta mais cedo, com quase a mesma qualidade.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** a versão de ~4 bits da Unsloth, entre IQ3_S e
  UD-Q4_K_XL em qualidade. Um download de 94 GB. Com menos de ~80 GB de RAM, o PolyStrata lê uma parte dele do SSD
  enquanto responde, então ali ele fica mais lento (um SSD NVMe ajuda).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): o mais próximo do modelo
  completo. Mas o PolyStrata lê a maior parte dele do SSD enquanto responde, então ele escreve só 7-8.5 tokens/s num PC
  com 64 GB.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** você configura à mão. Ele
  não está no menu do instalador.

Tamanhos, downloads e o que cabe onde: [docs/MODELS.md](docs/MODELS.md). Para adicionar outro modelo depois, rode
`SETUP.bat` (Linux: `./setup.sh --setup`).

### GLM 5.3 Flash

Três tamanhos por enquanto. O instalador escolhe o primeiro para o qual a sua RAM é suficiente.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 32 GB |
| --- | --- | ---: | ---: |
| **3.0BIT** | o modelo inteiro, a cerca de 3 bits ([neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)) | 114 GB | 96 GB |
| **UD-IQ2_XXS** | o modelo inteiro, a cerca de 2 bits ([Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)): lê um prompt mais rápido e escreve mais devagar | 102 GB | 82 GB |
| **REAP50-Q3_K_M** | metade dos experts roteados removida ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF), de patrickbdevaney) | 79 GB | 60 GB |

A RAM necessária depende da sua placa de vídeo: os experts que não cabem na placa precisam ficar na RAM.

| Sua placa de vídeo | 3.0BIT | UD-IQ2_XXS | REAP50-Q3_K_M | |
| --- | ---: | ---: | ---: | --- |
| **32 GB** | **96 GB** | **82 GB** | **60 GB** | o PC em que ele rodou (120 GB de RAM) |
| **24 GB** | cerca de 104 GB | cerca de 90 GB | cerca de 68 GB | ainda não testado |
| **16 GB** | cerca de 113 GB | cerca de 99 GB | cerca de 77 GB | ainda não testado |
| **12 GB ou menos** | | | | não inicia |

Os números de RAM são a estimativa do instalador, a partir do único PC em que ele rodou. Com menos RAM ele inicia
mesmo assim, mas lê experts do SSD enquanto responde e fica muito mais lento. O instalador avisa e pergunta antes.

As outras quantizações do GLM 5.3 Flash ainda não são suportadas: [o que falta](docs/GLM.md#what-is-missing) (em inglês).

### Qwen3.6-35B-A3B

Um tamanho por enquanto.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | o modelo inteiro com experts de 3 bits ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 17 GB | cerca de 21 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: cerca de 17 GB ao lado de uma placa de 12 GB, 13 GB ao lado de uma de 16 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

### Ornith 1.5

Um tamanho por enquanto.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **APEX-I-COMPACT** | o modelo inteiro com experts de 3 e 4 bits ([APEX](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF), de mudler) | 17 GB | cerca de 22 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: cerca de 17 GB ao lado de uma placa de 12 GB, 13 GB ao lado de uma de 16 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

### Gemma 4 26B-A4B

Um tamanho por enquanto.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 8 GB |
| --- | --- | ---: | ---: |
| **UD-Q3_K_M** | o modelo inteiro com experts de 3 bits ([Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), e o modelo de rascunho do Google para ele | 13 GB | cerca de 18 GB |

Uma placa maior guarda mais do modelo e precisa de menos RAM: cerca de 14 GB ao lado de uma placa de 12 GB e 10 GB quando a placa guarda tudo. São estimativas do instalador. Elas não foram
conferidas num PC assim.

## Como usar

Os cinco modelos são usados do mesmo jeito: o mesmo app, a mesma API.

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="A aba Monitor do app PolyStrata ao lado de um agente de programação"><br>
<sub>O <b>Monitor</b> do app PolyStrata (à esquerda) enquanto um agente de programação escreve o jardim com pagode do vídeo (à direita). Uma captura de tela do <a href="https://github.com/Niko1221/Strata">Strata</a>.</sub></p>

- **No navegador:** abra `http://127.0.0.1:8080`. Lá tem o **Chat**, um **Monitor** ao vivo do modelo e da sua
  GPU/CPU/RAM, e o **About** com as configurações e os endereços.
- **Seus apps e agentes de programação:** adicione um provedor "OpenAI-compatible" com a URL base
  **`http://127.0.0.1:8080/v1`**. Qualquer chave de API e qualquer nome de modelo funcionam.
  - Apps que usam a API da Anthropic: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI e outros apps que usam a OpenAI Responses API: `/v1/responses`
    ([configuração](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Raciocínio:** escolha **off, low, medium ou high** no menu do chat ou no "reasoning effort" do seu app. Off é o
  mais rápido. High é o melhor para perguntas difíceis.
- **Do seu celular ou de outro PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Sempre defina uma chave.

Onde eles diferem:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4 26B-A4B |
| --- | --- | --- | --- |
| **Imagens** | Responda sim para "Images?" na instalação. Depois clique em **Picture** no chat, ou anexe imagens no seu app. Placas AMD entendem imagens no Linux usando o processador; no Windows, ainda não. | Ainda não. | Responda sim para "Images?" na instalação, e depois como no Qwen3.8-Flash-Next. Numa placa com menos de 16 GB, o codificador de imagens roda no processador, e uma imagem leva alguns segundos. |
| **Uma requisição por vez** | Por padrão, o PolyStrata responde uma requisição e as outras esperam. Para responder várias ao mesmo tempo, defina `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Numa placa de 12 GB, isso deixa cada resposta mais lenta. | Uma por vez, as outras esperam. | Uma por vez, as outras esperam. |
| **Várias conversas** | A última é mantida, então as mensagens seguintes começam em segundos. Dois argumentos do motor mantêm várias na RAM ([detalhes](docs/DETAILS.md#using-it)). | A última é mantida. A instalação pergunta se deve manter várias na RAM, para que um agente e seus subagentes se revezem sem que os prompts sejam lidos de novo ([como](docs/GLM.md#several-conversations-at-once)). | Como o GLM 5.3 Flash. |
| **Prompts longos** | O PolyStrata lê a primeira mensagem de uma conversa inteira, cerca de 1 minuto a cada 30,000 tokens. As mensagens seguintes começam em segundos. | A primeira mensagem é lida inteira, cerca de 1 minuto a cada 30,000 tokens. | A primeira mensagem é lida inteira, cerca de 10 segundos a cada 30,000 tokens. |

Mais: [onde suas conversas ficam salvas](docs/INSTALL.md#where-things-are-stored), [a API](docs/DETAILS.md#using-it).

## Algo deu errado?

- **Meu PC travou na primeira vez que o PolyStrata iniciou.** Isso é normal enquanto ele carrega o modelo. Espere e não
  feche a janela. Ainda travado depois de 10 minutos? Reinicie o PC, feche outros programas e tente de novo, ou
  escolha um tamanho menor.
- **Parou durante o download ou a instalação.** Rode `START-HERE.bat` (ou `./setup.sh`) de novo. Ele continua de
  onde parou.
- **Está muito lento e a luz do disco não para de piscar, ou aparece "the engine stopped unexpectedly".** Seu PC
  não tem RAM livre suficiente. Feche outros programas (navegadores usam muita), ou escolha um tamanho menor (Q2_0
  ou IQ2_XS).
- **Diz que a porta 8080 já está em uso.** O PolyStrata já está rodando. Procure a janela dele.

Mais problemas e como resolver: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Ainda sem solução? Abra uma
[issue](https://github.com/VecSzn/PolyStrata/issues) e anexe o `strata-<model>.log` da pasta do PolyStrata. Encontrou um
problema de segurança? Informe em particular: [SECURITY.md](SECURITY.md).

## Como funciona?

Modelos como este normalmente rodam em servidores com centenas de gigabytes de memória de vídeo. Sua placa de vídeo
tem 8-32 GB. O PolyStrata faz o modelo caber **dividindo o trabalho pelo PC inteiro**. Pense numa cozinha: o que você
usa o tempo todo fica na bancada, e o resto espera na despensa.

### Qwen3.8-Flash-Next

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Os 24,576 experts do modelo: os mais usados na placa de vídeo, todos na RAM, uma tabela de consulta no SSD"></p>

- **O modelo é uma equipe de 24,576 pequenos especialistas ("experts").** Cada palavra precisa de só 10 deles.
- **Sua placa de vídeo** guarda os poucos milhares de experts usados com mais frequência. **Sua RAM** guarda todos
  eles, e **seu processador** trabalha no resto ao mesmo tempo. **Seu SSD** guarda uma grande tabela de consulta.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Um pequeno ajudante adivinha as próximas palavras; o modelo grande confere todas de uma vez e fica com as certas"></p>

- **Adivinhar e conferir:** um pequeno ajudante adivinha as próximas palavras. O modelo grande confere todas de
  uma vez. Você recebe a mesma resposta, 1.6-1.8x mais cedo.
- **Textos longos são lidos em grandes pedaços** (até 8,192 tokens por vez), a mais de 1,000 tokens por segundo.

A explicação mais longa: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Cada parte e seus números:
[os detalhes](docs/DETAILS.md#how-it-works) e o [artigo](docs/paper/Strata-Paper.pdf).

### GLM 5.3 Flash

- **O modelo tem 12,096 experts**, 288 em cada uma de 42 camadas. Cada palavra precisa de 8 de cada camada.
- **Sua placa de vídeo** guarda os mais usados, cerca de 2,470 numa placa de 32 GB. O motor conta quais o seu
  trabalho usa e escolhe de novo na próxima inicialização.
- **Enquanto ele escreve**, seu processador calcula os experts que a placa não tem. **Enquanto ele lê um prompt**,
  eles passam pela placa, 1,024 tokens por vez.
- **Adivinhar e conferir:** um bloco do modelo feito para adivinhar propõe as próximas palavras, e o modelo confere
  todas de uma vez.
- **O arquivo do modelo é usado onde está**, sem uma segunda cópia no disco. Uma inicialização leva cerca de
  6 segundos quando o arquivo já está na sua RAM.

Cada parte e seus números: [docs/GLM.md](docs/GLM.md#how-it-uses-the-card) (em inglês).

### Qwen3.6-35B-A3B

- **O modelo tem 10,240 experts**, 256 em cada uma de 40 camadas. Cada palavra precisa de 8 de cada camada.
- **Sua placa de vídeo** guarda quantos couberem, todos a partir de 24 GB. Limitada a 8 GB, a placa de teste guardou
  cerca de 1,200, e eles atenderam mais ou menos metade do que uma resposta usou: enquanto ele escreve, o motor traz
  para a placa os experts que a resposta escolhe sem parar.
- **Adivinhar e conferir:** o bloco de rascunho do próprio modelo, que está no arquivo dele, propõe as próximas
  palavras, e o modelo confere todas de uma vez.
- **O arquivo do modelo é usado onde está**, sem uma segunda cópia no disco.

### Ornith 1.5

- **A arquitetura do Qwen3.6-35B-A3B com outros pesos:** 10,240 experts, 256 em cada uma de 40 camadas, e 8 de cada
  camada para cada palavra.
- **Sua placa de vídeo** guarda quantos couberem, todos a partir de 24 GB. Limitada a 8 GB, a placa de teste guardou
  cerca de 1,600, e eles atenderam mais ou menos metade do que uma resposta usou.
- **Adivinhar e conferir:** o bloco de rascunho dele também está no arquivo. Ele acerta menos vezes que o do
  Qwen3.6-35B-A3B.
- **O arquivo do modelo é usado onde está**, sem uma segunda cópia no disco.

### Gemma 4 26B-A4B

- **O modelo tem 3,840 experts**, 128 em cada uma de 30 camadas. Cada palavra precisa de 8 de cada camada, além de
  uma parte da camada que todas as palavras usam.
- **Sua placa de vídeo** guarda quantos couberem, todos a partir de 16 GB pela estimativa do instalador. Limitada a
  8 GB, a placa de teste guardou cerca de 200, e com os que o motor trouxe enquanto escrevia eles atenderam de um
  terço à metade do que uma resposta usou.
- **Adivinhar e conferir:** o pequeno modelo de rascunho do Google para ele, um arquivo de 0.5 GB que o instalador
  baixa junto com o modelo, propõe as próximas palavras.
- **O arquivo do modelo é usado onde está**, sem uma segunda cópia no disco.

Cada parte dos três e seus números: [docs/BACKENDS.md](docs/BACKENDS.md#how-a-backend-works) (em inglês).

## Créditos e licença

O Qwen3.8-Flash-Next é da [equipe Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Ele foi comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) e Unsloth.
O GLM 5.3 Flash roda a partir dos arquivos de [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF),
da Unsloth e de patrickbdevaney. O Qwen3.6-35B-A3B é da equipe Qwen e o Gemma 4 do Google, os dois na quantização da
[Unsloth](https://huggingface.co/unsloth). O Ornith 1.5 é da ornith-ai, na quantização APEX de
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF). O PolyStrata é feito sobre o [Strata](https://github.com/Niko1221/Strata) de Niko1221 e seus colaboradores,
e usa partes do [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos os créditos:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). O PolyStrata é de código aberto sob a [MIT License](LICENSE).
Algumas partes e todos os modelos têm licenças próprias ([quais](docs/HOW_IT_WORKS.md#license)).

## Apoiar

O PolyStrata é gratuito e de código aberto. A maior parte dele é trabalho do autor do
[Strata](https://github.com/Niko1221/Strata). Se ele é útil para você, você pode apoiar esse trabalho:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
