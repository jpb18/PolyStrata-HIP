# Auditoria de ruído do repositório — 13 de outubro de 2026

## Resultado e limites

O problema principal nas pesquisas é o histórico de benchmarks, não código obsoleto confirmado.
Os ambientes e builds ocupam a maior parte do espaço, mas já estavam excluídos das pesquisas normais pelo Git.
Foram aplicadas apenas regras de pesquisa, duas exclusões de ficheiros locais e orientação curta para agentes.
Não houve eliminações, movimentos, alterações ao índice do Git, commit, push, recompilação ou arranque de servidores.

Esta auditoria cobre a árvore local em `<repo>`, incluindo ficheiros ocultos e
ignorados. O inventário inicial contém 35 450 ficheiros, dos quais 2 707 rastreados pelo Git. Não foram encontrados
erros de leitura, symlinks ou junctions na enumeração. `engine/` existe, mas não contém ficheiros nesta árvore.
Modelos e packs fora do repositório, caches globais e instalações externas não estão incluídos.

Os tamanhos são a soma dos comprimentos dos ficheiros: MiB = 2^20 bytes; GiB = 2^30 bytes. Não medem blocos
efetivamente ocupados no disco e podem contar hardlinks várias vezes. As categorias abaixo não se sobrepõem.
São um retrato anterior às pequenas alterações desta auditoria; não representam espaço autorizado para libertar.

## Espaço por categoria

| Categoria | Ficheiros | Tamanho lógico | Tratamento |
| --- | ---: | ---: | --- |
| 1. Código, ferramentas, dados distribuídos e documentação de referência | 903 | 30,800 MiB | Manter acessível |
| 2. Binários operacionais, dependências HIP e backups de rollback | 19 456 | 13 768,856 MiB | Preservar |
| 3. Relatórios, evidências e ferramentas específicas de experiências | 2 557 | 45,141 MiB | Preservar; pesquisar de forma dirigida |
| 4. Artefactos regeneráveis e dependências reinstaláveis | 11 407 | 772,085 MiB | Ignorar nas pesquisas; remoção depende de autorização |
| 5. Estado local cuja retenção precisa de confirmação (`.lcm/`) | 3 | 11,439 MiB | Preservar e confirmar finalidade/retenção |
| Metadados Git, separados das cinco categorias | 1 124 | 28,857 MiB | Manter |

Total fora de `.git/`: **15 338 905 294 bytes = 14,285 GiB**, em 34 326 ficheiros.
Total incluindo `.git/`: 15 369 164 359 bytes. Zero bytes foram libertados.

Na categoria 2, os 13 286,803 MiB de `.rocm-win/` são contabilizados conservadoramente como dependências
operacionais/de compilação. Incluem bibliotecas e ferramentas reinstaláveis, mas estas são necessárias aos fluxos HIP
e não foi validado que possam ser retiradas separadamente. Esta categoria inclui também executáveis `strata*`, DLLs
e ficheiros `last-*` no topo dos builds, e as configurações locais `strata-*.json`.

A categoria 4 inclui `.venv/`, o checkout local de llama.cpp, caches Python e os restantes produtos de build,
incluindo executáveis de testes. A regeneração pode exigir rede, o commit fixado, ROCm e tempo de compilação.
Ser regenerável não significa que esteja sem uso. Na categoria 3, captures e ficheiros `tmp/` de experiências foram
mantidos como evidência conservadoramente; a localização ou extensão não autoriza descartá-los.

## Principais fontes de ruído

| Local | Ficheiros | Tamanho | Observação |
| --- | ---: | ---: | --- |
| `.rocm-win/` | 19 427 | 13 286,803 MiB | SDK/runtime HIP; ignorado pelo Git |
| `build-hip-poly-win/` | 482 | 686,267 MiB | Mistura operacional, rollback, testes e intermediários |
| `build-hip-poly-prefill/` | 160 | 219,439 MiB | Candidato e build da experiência de prefill |
| `.venv/` | 7 127 | 182,131 MiB | Ambiente Python necessário ao servidor e ferramentas |
| `third_party/llama.cpp-3cf0325/` | 3 608 | 164,273 MiB | Dependência fixada usada pelos dois builds HIP |
| `bench/results/` | 2 537 | 44,747 MiB | 1 802 rastreados; 13,438 MiB de histórico rastreado |
| `.lcm/` | 3 | 11,439 MiB | Base de dados de sessão, WAL e SHM |
| `AMD_*.md` na raiz | 12 | 0,348 MiB | Relatórios de implementação/validação; seis ainda não rastreados |
| `docs/benchmarks/` | 3 | 0,021 MiB | Evidência, não orientação geral de desenvolvimento |

Os logs fora de ambientes/dependências somam 312 ficheiros e 2,994 MiB; 21 logs são rastreados
(156 204 bytes). Os caches Python fora desses ambientes somam 61 ficheiros e 2,086 MiB. Estes números
são subconjuntos das categorias anteriores, não espaço adicional.

As maiores pastas de experiências são `2026-10-12-hip-decode-validation` (10,688 MiB),
`2026-10-12-hip-prefill` (8,316 MiB), `2026-10-12-hip-failclosed-validation` (3,921 MiB) e
`2026-10-13-hip-prefill-integration` (3,899 MiB). Parte do volume corresponde a bases de sessão, prompts,
respostas e traces. Não foi encontrada evidência suficiente para declarar estas pastas abandonadas.

### Binários duplicados com finalidade documentada

Foram comparados SHA-256 dos executáveis e DLLs dos builds, não apenas nomes, tamanhos ou datas:

- `build-hip-poly-win/strata-poly.exe`, `strata-poly-failclosed.exe`, `strata-poly-failclosed-updated.exe`
  e `strata-poly-pre-prefill-20261013.exe`: mesmo hash `5aa60e285444a282…`; operacional e cópias de recuperação.
- `strata-poly-pre-failclosed.exe` e `strata-poly-rollback-decode-20261012.exe`: mesmo hash `7feddc7ef0d1c398…`.
- `build-hip-poly-prefill/strata-poly.exe` e `build-hip-poly-win/strata-poly-prefill-recent-20261013.exe`:
  mesmo hash `0618205e4137237b…`; candidato preservado.
- `amdhip64_7.dll` e `amd_comgr.dll` têm hashes iguais entre os dois builds. Cada diretório precisa de uma
  resolução válida das dependências; substituir cópias por links ou retirá-las requer validação operacional.

O launcher aponta explicitamente para `build-hip-poly-win/strata-poly.exe`. O
[relatório de integração](../bench/results/2026-10-13-hip-prefill-integration/README.md) documenta que o candidato
de prefill foi integrado, medido e revertido por piorar o uso representativo. O
[relatório anterior](../bench/results/2026-10-12-hip-prefill/README.md) descreve ganhos noutros cenários.
São etapas diferentes, não relatórios redundantes: ler ambos para interpretar o resultado. Preservar candidato,
operacional, backups, configurações em `rollback/` e patches, incluindo os ignorados pelo Git.

## Ficheiros rastreados e regras de ignore

**Não foram encontrados `.exe`, `.dll`, `.obj`, `.o`, `.lib` ou `.pyc` rastreados**, nem ficheiros rastreados
nos builds, ambientes ou `.lcm/`. Não há uma lista comprovada de produtos de compilação a retirar do índice.

| Ficheiros rastreados que merecem atenção | Decisão |
| --- | --- |
| 1 802 ficheiros de `bench/results/`, incluindo 179 scripts/fontes de experiências na árvore local completa | Preservar dados e reprodutores; excluir do âmbito de pesquisa habitual |
| Logs de validação, por exemplo `2026-09-30-community-rtx-5090/engine.log`, `2026-10-05-community-gfx1151/engine-{A,B,C,D,E}.log` e `2026-10-11-hip-qwen36-q3/logs/*` | Evidência intencional; não executar `git rm --cached` só por corresponderem a regras de ignore |
| `2026-10-07-community-xtx-6800xt-1m-helper/build-recipes/{baseline-hip,tuned-hip}.sh` | Receitas históricas legítimas; o padrão amplo `build*/` também as apanha |
| `data/draft_vocab*.bin`, `data/expert-profile*.bin` | Dados distribuídos referenciados por `setup.py` e ferramentas; manter no Git |
| `docs/media/*`, `docs/paper/Strata-Paper.pdf` | Recursos de documentação; ser gerado ou binário não basta para ignorar |
| Seis relatórios AMD rastreados na raiz e três ficheiros de `docs/benchmarks/` | Organização por navegação/pesquisa agora; eventual mudança de localização exige autorização |

`.gitignore` não retira ficheiros já rastreados. A regra global `*.log` e a regra `logs/` já ocultavam novos logs
de evidência, e `build*/` também corresponde a diretórios de receitas. Há exceções locais em vários resultados de
12/13 de outubro para conservar logs compactos. Recomenda-se manter esse mecanismo por experiência, com exceções
explícitas para evidência relevante, em vez de ignorar toda a árvore de resultados ou permitir todos os dumps.

Não foi alterado `bench/micro/*.txt`: pode esconder futuros inputs de testes, mas não existe `bench/micro/`
nesta árvore para determinar uma alternativa segura. Também não foram alargados padrões globais para `*.bin`,
`*.json`, `*.patch` ou `*.txt`, pois ocultariam dados necessários e evidência.

## Documentação e ferramentas a manter ou confirmar

- Entradas de desenvolvimento: `CMakeLists.txt`, `cmake/`, `src/`, `include/`, `sycl/`, `serve/`, `tools/`,
  `tests/`, `setup.py`, launchers e scripts Docker. `sycl/` contém um backend ativo, não uma cópia descartável.
  `third_party/ggml/` contém três ficheiros rastreados necessários e continua visível.
- `ref/load.py` continua necessário: `tools/canonical_xcheck.py` importa `_dequant_flat` desse ficheiro.
  Não deve ser removido por parecer uma implementação de referência antiga.
- `tools/hip/poly_api_check.py`, `bench_prefill.py` e `test_gpu_failure.py` têm referências em documentação ou
  validações recentes. `tools/hip/decode_trace.py` é trabalho local não rastreado. Manter todos.
- Scripts como `bench/results/2026-10-12-hip-prefill/{api_check,correctness,prefill_bench}.py` e
  `2026-10-13-hip-prefill-integration/{replay_ab,logproxy,server_ctl}.py` são reprodutores de experiências.
  Confirmar com o responsável se algum deve tornar-se ferramenta mantida em `tools/`; não mover nesta auditoria.
- `tools/hip/kv_summary.py` e `poly_serve_timing.py` são ferramentas manuais específicas. Não ter integração
  automática não prova desuso; confirmar proprietário e uso antes de qualquer retirada.
- `.lcm/lcm.db`, `lcm.db-wal` e `lcm.db-shm`: estado de sessão, não código. Já eram excluídos localmente.
  A finalidade e política de retenção precisam de confirmação; nunca remover um WAL de uma base ativa.

Para navegação, usar [INSTALL.md](INSTALL.md) e [AI_SETUP.md](AI_SETUP.md) para instalação humana e assistida,
[BACKENDS.md](BACKENDS.md) para modelos/backends, [AMD_HIP.md](AMD_HIP.md) para HIP e
[DETAILS.md](DETAILS.md) para o motor/API/configurações. Os README traduzidos destinam-se a outros idiomas,
e INSTALL/AI_SETUP destinam-se a públicos diferentes; não há fundamento para os eliminar como duplicados.

O `AGENTS.md` principal tinha uma descrição do segundo motor limitada a Linux/NVIDIA, incompatível com as
instruções Windows de BACKENDS e o trabalho HIP experimental documentado. Foi corrigida para remeter para esses
documentos. Os `AGENTS.md` e `CLAUDE.md` do checkout llama.cpp são instruções dessa dependência, não pontos
de entrada do projeto; continuar a excluí-los da navegação habitual.

## Alterações seguras aplicadas

1. Nova `.rgignore`: omite `/bench/results/`, `/docs/benchmarks/` e `/AMD_*.md` das pesquisas padrão do ripgrep.
   Não altera tracking, conteúdo ou funcionamento. As pesquisas dirigidas continuam disponíveis:

   ```sh
   rg --no-ignore "prompt-cache-recent" bench/results/2026-10-13-hip-prefill-integration
   rg --no-ignore "strata-poly" AMD_HIP_DECODE_VALIDATION.md
   ```

2. `.gitignore`: acrescenta `/.lcm/` e `/third_party/llama.cpp-*/`, antes dependentes de `.git/info/exclude`
   nesta máquina. O cache dos dois builds confirma `STRATA_GGML_DIR=third_party/llama.cpp-3cf0325`.
   Mantém visível o ggml distribuído. Foi removida apenas a recomendação de regenerar fixtures com
   `tools/make_tiny_model.py`, pois esse script não existe nesta árvore; os padrões das fixtures permanecem.
3. `AGENTS.md`: dois tópicos curtos indicam onde pesquisar, como consultar o histórico e o que preservar.
   As instruções de contribuição e validação existentes continuam presentes.

Impacto medido: `rg --files` passa de **3 105 para 894 ficheiros** após incluir este relatório, cerca de **71 % menos**.
Saem da listagem 2 197 ficheiros de resultados, os 12 relatórios AMD e os três benchmarks de docs.
A contagem mede nomes de ficheiros, não tokens nem duração de pesquisas. Outros agentes que usem `git ls-files`,
leitura direta ou ferramentas que não respeitem `.rgignore` precisam de seguir o âmbito indicado em `AGENTS.md`.

Risco destas alterações: uma pesquisa ampla pode deixar de mostrar evidência histórica relevante.
A recuperação é explícita e reversível com `rg --no-ignore`, sempre limitado à pasta/ficheiro relevante.
Não usar essa opção na raiz sem necessidade, pois incluiria ambientes, builds e dependências.

## Proposta de limpeza posterior

| Ação proposta | Impacto potencial | Risco e condição |
| --- | --- | --- |
| Manter as regras de pesquisa e consultar READMEs por experiência | Menos contexto gasto; sem espaço libertado | Baixo; já aplicado |
| Acrescentar exceções locais para logs/receitas que se decidam publicar como evidência | Evita ocultar novos relatórios por acidente | Baixo, mas confirmar a lista; não aplicado |
| Atualizar índices/links que apontem para resultados anteriores a uma integração/reversão | Evita interpretar resultados parciais como estado operacional | Baixo após rever referências; preservar originais |
| Retirar apenas intermediários e testes compilados dos builds | Até 423,659 MiB segundo a classificação, antes de rever cada ficheiro | Exige autorização; pode impedir rebuild incremental e diagnóstico. Não apagar as pastas completas |
| Recriar `.venv/` ou o checkout fixado llama.cpp | 182,131 / 164,273 MiB | Exige autorização e garantia de reinstalação; ambos estão em uso |
| Reduzir/reinstalar componentes ROCm | Maior volume potencial, ainda não quantificado como removível | Alto; bibliotecas e compiladores operacionais, sem validação de um pacote mínimo |
| Deduplicar backups ou DLLs dos builds | Duplicações identificadas por hash | Não recomendado nesta auditoria: backups e autonomia dos builds são deliberados e devem ser preservados |
| Expirar estado de sessão ou dumps de experiências | Pequeno face ao SDK | Exige confirmação de retenção, preservação da evidência e autorização explícita |
| Mover relatórios AMD da raiz ou históricos para arquivo | Melhora navegação sem reduzir bytes | Reorganização estrutural; exige autorização e atualização de todas as referências |
| Retirar qualquer ficheiro do índice do Git | Só após demonstrar que é um produto gerado sem valor de evidência | Exige aprovação concreta; nenhum candidato foi comprovado nesta auditoria |

As ações destrutivas acima são propostas, não pedidos implícitos de autorização. Relatórios de validação,
patches de rollback e binários operacionais/backups permanecem protegidos. O facto de estarem ignorados, serem
antigos ou partilharem um hash não muda essa decisão. Não foi encontrado código que possa ser declarado
obsoleto apenas pela análise estática desta auditoria.

## Verificação

Foram verificadas as regras com `git check-ignore`, a visibilidade dos diretórios de código com `rg --files`,
a consulta dirigida do histórico e a preservação dos ficheiros existentes. Os hashes e a lista inicial de
ficheiros permitem verificar alterações/eliminações no trabalho local e nos artefactos protegidos.
A verificação final passou: todos os ficheiros do inventário inicial continuam presentes, os tamanhos fora de
`.git/` estão inalterados salvo as edições autorizadas, e 2 708 hashes de ficheiros preservados coincidem.
Foram mantidas todas as entradas prévias do estado Git; as únicas novas entradas são `.gitignore`, `AGENTS.md`,
`.rgignore` e este relatório. A comparação de hashes cobre código previamente modificado, evidência (incluindo
ficheiros ignorados), executáveis/DLLs dos builds, perfis locais e patches; para os ambientes e dependências
grandes foi verificada a presença e o tamanho, não o conteúdo integral por hash.
Nenhum backend foi compilado: não foi alterado código de execução,
configuração de modelos ou build. Estes checks não substituem uma validação operacional de futuras limpezas.
