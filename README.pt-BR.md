<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

<p align="center"><b>Rode modelos de IA grandes no seu próprio PC</b><br>
Qwen3.8-Flash-Next e GLM 5.3 Flash · gratuito e de código aberto</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Um jardim com pagode em voxels que o modelo do PolyStrata escreveu, rodando no navegador"></a><br>
<sub>Um jardim com pagode em voxels, feito com um único prompt numa RTX 5070 com o <a href="https://github.com/Niko1221/Strata">Strata</a> (Qwen3.8-Flash-Next IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

O PolyStrata roda modelos de IA grandes num PC comum, modelos que normalmente precisam de um servidor.
Eles conversam, escrevem código e funcionam com seus apps e agentes de programação. Nada sai do seu PC.

Ele roda dois modelos, cada um com seu próprio motor:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** numa placa de vídeo NVIDIA ou AMD de
  12 GB ou mais, no Windows ou no Linux. Ele também entende imagens. O motor dele é o do
  [Strata](https://github.com/Niko1221/Strata), sem mudanças.
- **[GLM 5.3 Flash](docs/GLM.md)** no Linux com uma placa NVIDIA. O motor dele foi escrito para o PolyStrata. É
  experimental: só rodou em um PC, com uma placa de 32 GB.

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

Medido para o PolyStrata numa estação de trabalho, com um prompt de 29K tokens.

<table>
<tr><th>NVIDIA: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 120 GB de RAM</th></tr>
<tr><td>

| Tamanho | Escreve respostas | Lê seu prompt |
| --- | ---: | ---: |
| **3.0BIT** | 27 tokens/s | 312 tokens/s |
| **REAP50-Q3_K_M** | 27 tokens/s | 384 tokens/s |

</td></tr>
</table>

No mesmo PC, o llama.cpp escreveu 26 tokens/s com o arquivo 3.0BIT e leu cerca de 265. O GLM 5.3 Flash é um modelo
muito maior que o Qwen3.8-Flash-Next, e só cerca de um quinto dos experts dele cabe nos 32 GB desta placa, por isso
ele escreve mais devagar. O que foi medido e como: [docs/GLM.md](docs/GLM.md#what-it-was-measured-on) (em inglês).

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
| **RAM** | Ao lado de uma placa de 32 GB, **96 GB ou mais** para o modelo inteiro, 60 GB para o tamanho com metade dos experts. Uma placa menor precisa de mais RAM ([quanto](#qual-modelo-devo-escolher)). Ele rodou com 120 GB. |
| **Disco** | 114 GB ou 79 GB para o arquivo do modelo, num SSD. |
| **Sistema** | Linux e um driver NVIDIA atual. |

O instalador cuida de todo o resto. Ele compila o motor no seu PC e oferece instalar o que falta para isso
(g++ e o CUDA Toolkit). Ainda não: Windows, placas AMD, várias placas, imagens.

A lista completa: [docs/GLM.md](docs/GLM.md#what-you-need) (em inglês).

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
- se ele deve entender imagens (Qwen3.8-Flash-Next).

Aperte Enter em cada uma para usar a resposta recomendada. Depois ele baixa o modelo (cerca de 70 GB para o
Qwen3.8-Flash-Next, 79 ou 114 GB para o GLM 5.3 Flash) e o inicia.
Se o download parar, rode de novo: ele continua de onde parou. Seu navegador abre o app do PolyStrata em
`http://127.0.0.1:8080`.

O GLM 5.3 Flash está na lista de modelos do instalador. `./setup.sh --family glm` vai direto para ele
([o que ele pergunta e grava](docs/GLM.md#install), em inglês).

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

Dois tamanhos por enquanto. O instalador escolhe o primeiro para o qual a sua RAM é suficiente.

| Tamanho | O que é | Download | RAM ao lado de uma placa de 32 GB |
| --- | --- | ---: | ---: |
| **3.0BIT** | o modelo inteiro | 114 GB | 96 GB |
| **REAP50-Q3_K_M** | metade dos experts roteados removida ([REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF), de patrickbdevaney) | 79 GB | 60 GB |

A RAM necessária depende da sua placa de vídeo: os experts que não cabem na placa precisam ficar na RAM.

| Sua placa de vídeo | 3.0BIT | REAP50-Q3_K_M | |
| --- | ---: | ---: | --- |
| **32 GB** | **96 GB** | **60 GB** | o PC em que ele rodou (120 GB de RAM) |
| **24 GB** | cerca de 104 GB | cerca de 68 GB | ainda não testado |
| **16 GB** | cerca de 113 GB | cerca de 77 GB | ainda não testado |
| **12 GB ou menos** | | | não inicia |

Os números de RAM são a estimativa do instalador, a partir do único PC em que ele rodou. Com menos RAM ele inicia
mesmo assim, mas lê experts do SSD enquanto responde e fica muito mais lento. O instalador avisa e pergunta antes.

As outras quantizações do GLM 5.3 Flash ainda não são suportadas: [o que falta](docs/GLM.md#what-is-missing) (em inglês).

## Como usar

Os dois modelos são usados do mesmo jeito: o mesmo app, a mesma API.

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

Onde os dois diferem:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash |
| --- | --- | --- |
| **Imagens** | Responda sim para "Images?" na instalação. Depois clique em **Picture** no chat, ou anexe imagens no seu app. Placas AMD entendem imagens no Linux usando o processador; no Windows, ainda não. | Ainda não. |
| **Uma requisição por vez** | Por padrão, o PolyStrata responde uma requisição e as outras esperam. Para responder várias ao mesmo tempo, defina `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Numa placa de 12 GB, isso deixa cada resposta mais lenta. | Uma por vez, as outras esperam. |
| **Várias conversas** | A última é mantida, então as mensagens seguintes começam em segundos. Dois argumentos do motor mantêm várias na RAM ([detalhes](docs/DETAILS.md#using-it)). | A última é mantida. A instalação pergunta se deve manter várias na RAM, para que um agente e seus subagentes se revezem sem que os prompts sejam lidos de novo ([como](docs/GLM.md#several-conversations-at-once)). |
| **Prompts longos** | O PolyStrata lê a primeira mensagem de uma conversa inteira, cerca de 1 minuto a cada 30,000 tokens. As mensagens seguintes começam em segundos. | A primeira mensagem é lida inteira, cerca de 1,5 minuto a cada 30,000 tokens. |

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
tem 12-32 GB. O PolyStrata faz o modelo caber **dividindo o trabalho pelo PC inteiro**. Pense numa cozinha: o que você
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
- **Sua placa de vídeo** guarda os mais usados, cerca de 2,570 numa placa de 32 GB. O motor conta quais o seu
  trabalho usa e escolhe de novo na próxima inicialização.
- **Enquanto ele escreve**, seu processador calcula os experts que a placa não tem. **Enquanto ele lê um prompt**,
  eles passam pela placa, 1,024 tokens por vez.
- **O arquivo do modelo é usado onde está**, sem uma segunda cópia no disco. Uma inicialização leva cerca de
  6 segundos quando o arquivo já está na sua RAM.

Cada parte e seus números: [docs/GLM.md](docs/GLM.md#how-it-uses-the-card) (em inglês).

## Créditos e licença

O Qwen3.8-Flash-Next é da [equipe Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Ele foi comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) e Unsloth.
O GLM 5.3 Flash roda a partir dos arquivos de [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)
e de patrickbdevaney. O PolyStrata é feito sobre o [Strata](https://github.com/Niko1221/Strata) de Niko1221 e seus colaboradores,
e usa partes do [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos os créditos:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). O PolyStrata é de código aberto sob a [MIT License](LICENSE).
Algumas partes e todos os modelos têm licenças próprias ([quais](docs/HOW_IT_WORKS.md#license)).

## Apoiar

O PolyStrata é gratuito e de código aberto. A maior parte dele é trabalho do autor do
[Strata](https://github.com/Niko1221/Strata). Se ele é útil para você, você pode apoiar esse trabalho:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
