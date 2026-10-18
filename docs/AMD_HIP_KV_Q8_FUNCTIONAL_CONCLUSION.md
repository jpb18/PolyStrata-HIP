# KV Q8 — conclusão funcional dos incidentes T5

**Estado: MANUALLY APPROVED pelo utilizador em 17/10/2026.**
Âmbito: interpretação e impacto funcional das diferenças de texto nos casos T5 de português
e transação, em FP16 e Q8. Não constitui aprovação geral do Q8 nem dispensa outros testes.

## Conclusões registadas

1. **A resposta inadequada já ocorre em FP16.** No caso português, ambos desviam uma pergunta
   sobre índices de bases de dados para caches KV/experts. No caso da transação, ambos recusam
   responder, alegando falta de informação nos documentos fornecidos. Q8 não é necessário
   para esses erros ocorrerem.
2. **O fator provável é a interpretação da tarefa pelo modelo sob influência do contexto.**
   Os prompts contêm 30720 e 32000 tokens, maioritariamente documentos sobre PolyStrata, antes
   da pergunta. As respostas sugerem que o modelo lhes atribuiu peso excessivo ou os tratou
   como restrição temática. O desenho do prompt pode contribuir. Esta é uma hipótese apoiada
   pelo comportamento observado, não uma causa isolada por experiência controlada.
3. **Q8 muda a formulação e os detalhes selecionados.** Exemplos incluem “geração de tokens”
   versus “leitura/geração” e uma explicação com tokens/s versus outra com GB/s. As unidades
   descrevem grandezas diferentes e podem alterar a impressão do leitor sobre desempenho.
   Esta nota não certifica os números produzidos pelo modelo.
4. **Não foi demonstrada uma falha funcional adicional Q8 nestes incidentes.** O desvio da
   tarefa e a recusa final são partilhados pelos dois formatos. Isto não prova equivalência
   geral, ausência de outros defeitos ou causalidade exclusiva do contexto longo.
5. **A qualidade geral das respostas é fraca para as tarefas pedidas.** A apresentação é
   organizada, mas o conteúdo não resolve a pergunta sobre índices nem descreve transferência,
   rollback e concorrência. A principal limitação observada é seguir a tarefa em presença
   dos documentos, não a diferença de palavras entre formatos.

## Decisão de fecho

O utilizador pediu para terminar este ponto, considerá-lo fechado e anotar as conclusões.
Fica **encerrada a análise funcional destes dois incidentes**, com a incerteza causal acima
declarada e aceite no âmbito deste fecho. Não há nova investigação ou inferência pendente
para este ponto. As quatro revisões por fingerprint T5 já estavam concluídas.

Reabrir se surgir uma falha funcional relevante e reproduzível em Q8 que não ocorra em FP16
sob condições comparáveis, ou nova evidência que contradiga estas conclusões.
Comparar contextos sem documentos/curtos/originais poderia testar a hipótese causal, mas
fica como opção futura, não como requisito pendente deste fecho.

## Estados que permanecem separados

- **T5 estrito:** FAIL; quatro divergências revistas, zero pareceres pendentes, NON_BLOCKING.
- **T4 v2 — aceitação atual: MANUALLY APPROVED.** O utilizador aceitou a exceção prosa-1K;
  T4 deixa de bloquear por decisão manual. O cálculo automático preservado é 14/15 PASS,
  com prosa-1K acima da regra numérica: ΔNLL + 2 EP = 0,02825556 > 0,02.
  Os exemplos T5 analisados aqui não são a resposta do caso prosa-1K e não explicam nem
  anulam o desvio numérico. Nenhuma consequência funcional específica de prosa-1K foi
  demonstrada por esta discussão.
- **T7, T9 e T11:** não foram alterados ou reexecutados por esta decisão.
- **Promoção definitiva Q8:** NOT YET APPROVED. Não se alteraram critérios, defaults ou motor.

## Evidência

- [Pareceres dos quatro fingerprints e limites da análise](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/T5-REVIEW.md).
- Respostas portuguesas completas: [FP16](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/outputs/portuguese-full-fp16-fit0.txt)
  e [Q8](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/outputs/portuguese-full-q8-fit0.txt).
- Respostas completas de transação: [FP16](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/probe/transaction-full-fp16-fit0/answer.txt)
  e [Q8](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/probe/transaction-full-q8-fit0/answer.txt).
  As [quatro continuações](../bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/probe-results.json)
  reproduziram os prefixos históricos e terminaram naturalmente; os tokens completos
  coincidiram entre FIT=0/1 dentro de cada formato.
- [Resultado T4 sob v2](../bench/results/2026-10-17-hip-kv-q8-t4-v2/README.md).
- [Registo estruturado deste fecho](../bench/results/2026-10-17-hip-kv-q8-functional-conclusion/DECISION.json).
- [Aprovação manual posterior, que define o estado atual](AMD_HIP_KV_Q8_MANUAL_APPROVAL.md).

Nota de engenharia registada por Codex a pedido do utilizador. Nenhuma nova inferência,
alteração de medições históricas ou encerramento de processos foi necessário. O registo
original de fecho permanece como fotografia anterior; a aprovação manual posterior substitui
o seu estado de aceitação T4 sem substituir os resultados automáticos.
