# KV Q8 — aprovações manuais T4 v2 e T7

**Estado operacional, 18/10/2026:** Q8 promovido para uso local por escolha;
FP16 mantém-se por omissão. T4/T5/semelhança/T7/T9/T11 aceites manualmente.
O âmbito e os resultados históricos abaixo mantêm as datas das respetivas decisões.
Referências antigas a NOT YET APPROVED foram substituídas pela
[promoção atual](../bench/results/2026-10-18-hip-kv-q8-promotion/README.md).
Ver [arranque e limites](HIP_LAUNCH_KV.md).


**Estado atual de aceitação T4 v2: MANUALLY APPROVED.**
Decisão do utilizador em 17/10/2026: “Considera antes "MANUALLY APPROVED" e não "FAIL"”.
O ponto de revisão das diferenças de texto fica também registado como MANUALLY APPROVED.

## Efeito da decisão

T4 deixa de bloquear a promoção por **aprovação manual explícita**, incluindo a exceção
de prosa-1K. Este é o estado a usar nos resumos atuais. A aprovação não é um PASS calculado
pelo avaliador e não altera a fórmula nem as medições.

| Dimensão | Registo |
|---|---|
| Aceitação atual T4 v2 | **MANUALLY APPROVED** |
| Impacto do gate T4 | NON_BLOCKING_BY_MANUAL_APPROVAL |
| Resultado automático preservado | 14/15 PASS; prosa-1K fora da regra numérica |
| Exceção aceite | ΔNLL + 2 EP = 0,02825556, margem 0,02 |
| Maior ΔNLL médio medido | +0,01407628 nats/token, prosa-1K |
| Alteração de fórmula ou dados | Nenhuma |

O [registo estruturado da aprovação](../bench/results/2026-10-17-hip-kv-q8-manual-approval/APPROVAL.json)
substitui o estado de aceitação dos resumos anteriores. Os seus FAIL automáticos permanecem
como resultados históricos do cálculo, associados às respetivas versões. Não se reescrevem
para fabricar um resultado de teste diferente.

## Conclusões funcionais aceites

Nos dois incidentes T5 revistos, a resposta inadequada já ocorre em FP16. Não foi demonstrada
uma falha funcional adicional causada pelo Q8. A interpretação da tarefa pelo modelo sob
influência dos documentos longos é o fator provável, sem causa isolada experimentalmente.
As diferenças de formulação e de detalhes não inverteram a conclusão principal nesses casos.

Esses exemplos T5 são distintos de prosa-1K: a sua revisão não é uma prova causal sobre
o caso numérico T4. A aceitação da exceção T4 é uma decisão manual do utilizador, registada
separadamente dessa análise funcional.

## T7 — custo e incerteza de desempenho aceites

**T7: CLOSED / MANUALLY APPROVED**, por decisão explícita do utilizador em 17/10/2026:
“sao aceitaveis”. Aceites o custo e a incerteza documentados na campanha B-8K de
40 execuções: velocidade Q8 −1,89% no adaptativo, IC 95% −5,07% a +1,38%, e
−9,91% na janela fixa experimental. O gate passa a NON_BLOCKING_BY_MANUAL_APPROVAL.
O INCONCLUSIVE automático do modo adaptativo permanece como resultado medido.
Não há nova margem universal nem alteração de resultados históricos.

[Registo posterior da aprovação T7](../bench/results/2026-10-17-hip-kv-q8-t7-manual-approval/README.md)
e [decisão estruturada](../bench/results/2026-10-17-hip-kv-q8-t7-manual-approval/APPROVAL.json).
As referências anteriores a T7 pendente descrevem estados anteriores a esta decisão.

## Restante validação

Atualização de 18/10/2026: os quatro controlos dirigidos de restauro a 261K
(FP16/Q8, sem MTP e janela fixa 4) estão completos, com tokens/logits iguais ao bit
dentro de cada formato. O FP16 com MTP passou com guarda runtime de RAM de 2,5 GiB;
os três anteriores usaram 4 GiB. RAM livre mínima nesta repetição: 3790,11 MiB.
[Resultado e limites](../bench/results/2026-10-17-hip-kv-q8-t9-long-context/FOLLOWUP_RAM2560.md).

As tentativas bloqueadas, o FAIL estrito de contagens da leitura nova Q8 e a divergência
adaptativa FP16 sem restauro permanecem preservados. A matriz histórica completa não
foi repetida. T11 está CLOSED / MANUALLY APPROVED por decisão do utilizador em 18/10/2026;
T9 está igualmente MANUALLY APPROVED; a promoção geral Q8 é uma decisão separada.


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
