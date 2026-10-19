# Escolher KV no launcher HIP

O ponto de entrada principal é `START_HIP_POLY.bat`. Os pré-requisitos e a compilação são passos explícitos no [README](../README.md); o launcher não instala dependências nem compila automaticamente.

FP16 continua por omissão. A opção Q8 foi aprovada para uso local por escolha em 18/10/2026, no âmbito documentado: Windows HIP, RX 9060 XT 16 GB, 32 GB de RAM e Qwen3.6-35B-A3B UD-Q3_K_M. Esta decisão não valida outras placas, modelos ou binários, nem estabelece igualdade de texto FP16/Q8. Consulte as [decisões manuais](AMD_HIP_KV_Q8_MANUAL_APPROVAL.md) e o [registo da promoção](../bench/results/2026-10-18-hip-kv-q8-promotion/README.md).

## Seleção e configuração

Sem argumentos, a sequência é formato KV → modelo → contexto → tecto de experts (Q8). Enter, EOF ou timeout na escolha de KV mantém FP16. Q8 não se torna uma preferência persistente. O menu FP16 lista os modelos configurados; Q8 lista apenas a configuração textual validada e a sua variante Vision.

```bat
START_HIP_POLY.bat
START_HIP_POLY.bat fp16 128k
START_HIP_POLY.bat q8_0 48k
START_HIP_POLY.bat q8_0 128k
START_HIP_POLY.bat q8_0 262k
START_HIP_POLY.bat q8_0 128k nocap
START_HIP_POLY.bat q8_0 check
```

Um modelo, GGUF ou `POLY_CONFIG` explícito dispensa o menu de modelos. Para FP16, `POLY_CONFIG` usa diretamente a configuração existente e preserva os seus argumentos. Sem ele, os perfis são 48K, 128K, 262K e 262K-stream. O launcher recusa substituir um perfil incompatível; escolha outro nome ou edite a configuração explicitamente. A ferramenta `poly_config.py` só substitui um ficheiro existente com `--overwrite` explícito.

`check` não inicia o motor. Em FP16, também não cria um perfil ausente. Em Q8, prepara um diretório temporário novo e verifica os artefactos/configuração/porta; não demonstra que há recursos suficientes para uma carga real. O arranque normal faz as verificações de recursos adicionais.

## Artefactos Q8

Este menu corre o mesmo motor dos outros menus, `build-hip-poly-win/strata-poly.exe`: há um único executável, sem binário Q8 separado nem hash fixado, e uma recompilação entra no arranque seguinte. O launcher mostra e regista o SHA-256 do motor que corre.

O perfil congelado incluído em `data/hip/qwen36-q3-q8-expert-profile.txt` tem SHA-256:

```text
1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d
```

A origem é a campanha Phase 4b; os bytes foram preservados. Cada arranque copia o perfil para o seu diretório privado, porque o motor atualiza contagens de utilização de experts.

A configuração base deve chamar-se `qwen3.6-35b-a3b-ud-q3_k_m-hip` ou a variante `-vision`, com backend HIP e o ficheiro `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. Estas verificações de nome/configuração não substituem verificar o SHA do GGUF: use o [registo do modelo medido](reports/AMD_QWEN36_Q3_MIGRATION.md#3-modelo-novo-e-verificação). Tokenizer, sampling, threads, porta e opções MTP vêm dessa configuração.

## Contexto, experts e guardas

Q8 oferece 32K (32,768), 48K (49,152), 128K (131,072) e 262K (262,144). Não aceita `262k-stream`. O percurso fonte mais geral e as diferenças numéricas de Q8 estão no [guia técnico](HIP_GUIDE.md#q8-kv-and-streaming).

| Contexto | Tecto medido, por omissão |
| --- | ---: |
| 32K / 48K | 5,000 experts |
| 128K | 3,500 experts |
| 262K | 2,000 experts |

Depois do contexto, Enter mantém esse tecto. A opção 2 ou `nocap` omite `--expert-cache-max`, deixando o motor dimensionar a cache à VRAM disponível. Os resultados a 262K e os limites da medição constam do [relatório de VRAM](reports/AMD_HIP_VRAM_FLOOR.md); os outros contextos sem tecto não foram medidos nesse relatório. Sem tecto, a verificação confirma uma cache não vazia e a contagem reportada, em vez de exigir o conjunto congelado de experts.

O perfil Q8 desativa swaps de experts e parking de conversas. As guardas de admissão exigem no máximo 3 GiB de VRAM de fundo e ausência de outra inferência incompatível. Depois da carga, exigem pelo menos 512 MiB de VRAM livre reportada e nenhuma falta da reserva reportada pelo piso de experts. Não há mínimo de RAM: o launcher só mostra a RAM livre. Num PC com 16 GB os experts fora da VRAM vêm das páginas do ficheiro, do disco quando não cabem em RAM, e o arranque e a velocidade nesse caso não foram medidos. O launcher confirma INFO, READY e saúde HTTP antes de anunciar o serviço.

O servidor usa `127.0.0.1`, na porta da configuração. Numa porta ocupada, o launcher pode consultar `/health`, mas não troca o modelo nem encerra outro servidor. Não há fallback automático. Configurações e logs novos ficam em `tmp/hip-launch-q8/q8-*`, incluindo `engine-q8_0.log` e `launcher-q8_0.log`.

Para voltar a FP16, encerre normalmente a sessão Q8 e escolha FP16/Enter no próximo arranque. Os ficheiros da configuração FP16 não são reescritos pela preparação Q8.

## Vision com Q8

Uma configuração Vision já instalada pode ser selecionada no menu Q8, ou explicitamente:

```bat
START_HIP_POLY.bat q8_0 qwen3.6-35b-a3b-ud-q3_k_m-hip-vision 48k
```

O launcher verifica a existência do encoder, projector e modelo e a correspondência do modelo textual. Preserva o encoder e a porta configurada; o motor Q8 recebe embeddings pelo protocolo Poly, sem uma opção CLI `--vision` adicional. A configuração FP16 e os seus executáveis não são substituídos.

O [registo de integração](../bench/results/2026-10-18-hip-q8-vision-integration/README.md) documenta um pedido real com uma imagem vermelha a 48K, resposta “Red” e encerramento normal. Esse smoke não é uma avaliação ampla de qualidade multimodal nem valida imagens com o contexto cheio. Os ficheiros brutos desses registos podem exigir o arquivo local de evidência.
