# Validação comparativa KV Q8 contra FP16

**Estado operacional, 18/10/2026:** Q8 promovido para uso local por escolha;
FP16 mantém-se por omissão. T4/T5/semelhança/T7/T9/T11 aceites manualmente.
O âmbito e os resultados históricos abaixo mantêm as datas das respetivas decisões.
Referências antigas a NOT YET APPROVED foram substituídas pela
[promoção atual](../bench/results/2026-10-18-hip-kv-q8-promotion/README.md).
Ver [arranque e limites](HIP_LAUNCH_KV.md).


Versão **KV-Q8-paired-v1-2026-10-17**, adotada a pedido do utilizador para focar os
testes na comparação Q8/FP16. Aplica-se às campanhas novas. A reanálise dos dados anteriores
fica identificada como retrospectiva e não altera resultados históricos.

## Controlos e decisão por tarefa

Mesmo modelo/pesos/tokenizer, prompt/instruções, executável, contexto, budgets, FIT/MTP
configurado, conjunto de experts, swaps, aquecimento e estado inicial das ferramentas.
A variável pretendida é a KV: FP16 contra Q8_0. Usam-se processos sequenciais, cópias do
perfil congelado e ordem FP16/Q8 e Q8/FP16 equilibrada. O servidor mantém as verificações
de executável/perfil/expert set e os limites de recursos. Configurações incompatíveis,
falhas nativas, valores não finitos, integridade e dados em falta não viram scores válidos.

| FP16 | Q8 | Classificação | Interpretação |
|---|---|---|---|
| PASS | PASS | BOTH_PASS | Ambos resolvem a tarefa |
| FAIL | FAIL | SHARED_TASK_FAILURE | Ambos falham; não demonstra regressão Q8 nem identifica a causa |
| PASS | FAIL | Q8_ONLY_FAILURE_OBSERVED | Candidato a regressão que requer revisão/repetição emparelhada |
| FAIL | PASS | Q8_ONLY_SUCCESS_OBSERVED | Ganho observado, sem prova de superioridade geral |

Os validadores absolutos de código, estado final, JSON e instruções mantêm-se. Os seus
PASS/FAIL por braço ficam guardados; acrescenta-se a decisão da comparação entre braços.
Ganhos noutras tarefas não podem ocultar falhas apenas do Q8. Uma discordância isolada
não basta para provar causa na quantização.

Estados comparativos:

- `ERROR`: integridade/configuração/infraestrutura inválida.
- `INCOMPLETE`: faltam resultados planeados; o plano define o denominador, não os sucessos.
- `REVIEW_REQUIRED`: há falha observada só no Q8, mesmo com empate na pontuação total.
- `BASELINE_LIMITED`: todos os pares falham nos dois braços; não declarar PASS.
- `NO_Q8_REGRESSION_OBSERVED`: sem os impedimentos anteriores e com um par informativo.
  Descreve a amostra; não certifica equivalência ou promoção.

Não se escolheu uma nova margem numérica a partir dos resultados. A margem antiga do T8
original (Q8 ≥ FP16−1 em 20 tarefas) fica como diagnóstico `legacy_original_t8`; não
substitui a revisão das discordâncias sob esta política.

## Papel dos testes

| Teste | Foco vigente |
|---|---|
| T1–T3 | Formato, kernels e erro numérico; mantidos |
| T4 | Probabilidades emparelhadas nos mesmos tokens; regra v2 e MANUALLY APPROVED mantidos |
| T5 | Diagnóstico de trajetória greedy; palavras iguais/gap não decidem qualidade |
| T6 | Recuperação das mesmas notas por par FP16/Q8 e plano completo |
| T7 | Eficiência MTP e desempenho separados; aceitação não mede qualidade |
| T8 | Validação real da tarefa por braço e comparação por tarefa/categoria |
| T9 | Integridade save/restore dentro de cada formato e comparação de sucesso operacional |
| T10 | Fail-closed em ambos os caminhos; requisito operacional mantido |
| T11 | FP16 do candidato contra FP16 operacional; proteção de integração mantida |

### T5

Os gaps 0,1/0,2 e a fração de trajetórias coincidentes permanecem métricas históricas e
diagnósticas. Divergência textual produz `DIAGNOSTIC_ONLY`, qualidade `NOT_EVALUATED`
até validação da tarefa/parecer específico. Outliers conservam fingerprint e revisão;
os quatro já revistos continuam revistos. Integridade/configuração/execução continuam
a impedir conclusões, independentemente de a divergência ser conhecida.

`kv_parity.py` mantém `parity` por omissão para streaming e verificações do mesmo formato.
Para T5 Q8 usar `--comparison-mode q8-relative`, referência FP16, teste Q8_0, cap fixo,
swaps 0 e o mesmo perfil. O modo cria cópias privadas do perfil e exige expert set igual.
Exit zero significa diagnóstico recolhido, não qualidade aprovada.

```text
python tools/hip/kv_parity.py --engine CANDIDATO --model MODELO --prompt IDS.txt
  --ctx 49152 --comparison-mode q8-relative --ref "--kv fp16" --test "--kv q8_0"
  --args "--expert-profile PERFIL --expert-cache-max 5000 --expert-swaps 0"
```

### T7

Confirmação de 17/10/2026: [40 execuções controladas em B-8K](../bench/results/2026-10-17-hip-kv-q8-t7-confirmation/RESULTS.md).
MTP adaptativo do candidato: duração Q8 +1,93%, IC 95% −1,37% a +5,34%,
**INCONCLUSIVE**; janela fixa experimental: +11,00%, IC +7,56% a +14,55%,
custo reproduzido nesta fixture. **Aceitação atual T7: CLOSED / MANUALLY APPROVED**
pelo utilizador em 17/10/2026, aceitando o custo e a incerteza documentados.
[Decisão posterior](../bench/results/2026-10-17-hip-kv-q8-t7-manual-approval/README.md):
T7 deixa de bloquear por aprovação manual. As medições, o INCONCLUSIVE automático,
os históricos abaixo e as aprovações funcionais anteriores mantêm-se.

Reportar aceites/ofertados e diferença em pontos percentuais sob controlos iguais.
A regra antiga −2 pp fica diagnóstica, sem produzir FAIL de qualidade. Texto diferente
muda o trabalho; mesmo com tokens iguais faltam schedules controlados e tempos calibrados
repetidos para concluir desempenho. Sem isso, `INCONCLUSIVE`. Não se declara uma queda
de aceitação inofensiva nem uma regressão de throughput só por essa métrica.

### T9–T11

Atualização de 18/10/2026: os quatro controlos dirigidos de restauro a 261K
(FP16/Q8, sem MTP e janela fixa 4) estão completos, com tokens/logits iguais ao bit
dentro de cada formato. O FP16 com MTP passou com guarda runtime de RAM de 2,5 GiB;
os três anteriores usaram 4 GiB. RAM livre mínima nesta repetição: 3790,11 MiB.
[Resultado e limites](../bench/results/2026-10-17-hip-kv-q8-t9-long-context/FOLLOWUP_RAM2560.md).

As tentativas bloqueadas, o FAIL estrito de contagens da leitura nova Q8 e a divergência
adaptativa FP16 sem restauro permanecem preservados. A matriz histórica completa não
foi repetida. T11 está CLOSED / MANUALLY APPROVED por decisão do utilizador em 18/10/2026;
T9 está igualmente MANUALLY APPROVED; a promoção geral Q8 é uma decisão separada.


## Ferramentas e reprodução

O [comparador](../tools/hip/functional_quality/comparison.py) é usado nos relatórios novos;
os freezes incluem a versão da avaliação e os testes. A pasta default é nova, preservando
a campanha antiga. Os oráculos das tarefas e as guardas de execução mantêm-se.

O [replayer](../tools/hip/kv_q8_replay.py) valida os hashes de planos, registos, transcritos
e logs disponíveis, sem modelo/GPU e com saída numa pasta nova fora das fontes históricas:

```powershell
.venv/Scripts/python.exe -B tools/hip/kv_q8_replay.py --out tmp/q8-paired-replay-new
```

Os resultados estão no [relatório de atualização](../bench/results/2026-10-17-hip-kv-q8-paired-tests/README.md).
C01..C14 partilhadas entre piloto e original são as mesmas observações, não replicações
independentes. Nenhuma decisão desta política promove automaticamente Q8 ou altera T4 manual.

## Aprovação manual T11 — 18/10/2026

**T11: CLOSED / MANUALLY APPROVED; NON_BLOCKING_BY_MANUAL_APPROVAL.**
O utilizador aceitou o tradeoff de latência documentado: leitura MTP +2,323%,
geração +1,138%, com tokens iguais nos seis pares. Sem MTP, logits iguais ao bit
e tempos dentro de ±2% nesta amostra. Resultados automáticos, dispersão e limitações
do observador MTP ficam preservados; não é uma nova margem universal nem promoção Q8.
[Decisão e evidência](../bench/results/2026-10-18-hip-t11-manual-approval/README.md).

## Aprovação manual T9 — 18/10/2026

**T9: CLOSED / MANUALLY APPROVED; NON_BLOCKING_BY_MANUAL_APPROVAL.**
Quatro controlos dirigidos de restauro a 261K completos; o utilizador aceita as
limitações adaptativas e de cobertura documentadas. O controlo Q8 adaptativo sem
restauro não foi executado. Não se apagam resultados automáticos nem se altera o motor.
T4, semelhança funcional, T7, T9 e T11 deixam de bloquear pelos motivos aceites.
A promoção operacional do candidato definitivo permanece separada; FP16 continua
a omissão e Q8 streaming não é validado por estas decisões.
[Decisão e situação para promoção](../bench/results/2026-10-18-hip-kv-q8-t9-manual-approval/README.md).
