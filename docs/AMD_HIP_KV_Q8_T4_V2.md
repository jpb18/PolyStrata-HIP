# Critério T4 v2 — adotado em 17/10/2026

**Estado de aceitação atual: [MANUALLY APPROVED](AMD_HIP_KV_Q8_MANUAL_APPROVAL.md).**
O utilizador aceitou explicitamente a exceção numérica prosa-1K. O resultado automático
14/15 e os seus valores permanecem na reanálise; não são apresentados como PASS automático.
Esta decisão encerra o bloqueio T4 por aprovação manual, mantendo a regra de cálculo abaixo.

Identificador: **T4-v2-2026-10-17**. Adoção explicitamente pedida pelo utilizador:
“Aplica essa mudança e corre o teste novamente, reporta os resultados de volta”.
Esta versão muda apenas a decisão T4 de fidelidade teacher-forced Q8 contra FP16.
Não altera motor, KV, executáveis, modelos, defaults, T5–T11 ou a decisão de promoção geral.

## Regra vigente

Para cada caso, nos mesmos 512 tokens forçados:

```text
delta[i] = log_p_fp16[i] - log_p_q8[i]
ΔNLL = média(delta)
EP = desvio-padrão amostral(delta) / sqrt(512)

PASS de NLL se ΔNLL + 2 × EP ≤ 0,02 nats/token
```

EP é não negativo. A condição implica também ΔNLL ≤0,02; não é necessário um segundo teste
redundante dessa média. O sentido é unilateral: o aumento de NLL é a perda que se limita.
Uma diferença negativa não falha só por estar longe de zero.

Mantêm-se os outros requisitos por caso:

| Contexto | KL média máxima | Concordância top-1 mínima |
|---|---:|---:|
| 1K | 0,03 | 95% |
| ≥8K | 0,15 | 90% |

Todos os casos têm de satisfazer NLL, KL e top-1. Uma métrica ausente/não finita, EP negativo,
amostra incompleta ou contexto desconhecido produz ERROR, nunca PASS.
O avaliador implementado está em [kv_q8_t4_policy.py](../tools/hip/kv_q8_t4_policy.py).

## Diferença face à v1

A regra anterior dizia “ΔNLL ≤ +0,02 e dentro de 2 erros-padrão”. A reconciliação v1
aplicou `abs(ΔNLL) ≤2 EP`, proximidade da diferença a zero. A v2 testa a diferença incluindo
uma margem de incerteza contra a tolerância de 0,02. Não aumenta o multiplicador EP nem
a tolerância de NLL: muda a referência dessa condição, conforme a proposta aprovada.

Pode, por isso, aceitar um caso v1 que falhava por ter uma diferença pequena e bem estimada
face a zero, mas rejeitar outro com mais incerteza perto de 0,02. Não foi definida para
garantir PASS de todos os resultados conhecidos.

O EP original continua a ser calculado por token. Tokens consecutivos e as fixtures escolhidas
podem ser correlacionados. `ΔNLL + 2 EP` é um limite descritivo para esta decisão de engenharia;
não se afirma cobertura estatística calibrada, equivalência funcional ou que um FAIL prove
degradação real superior a 0,02.

## Versionamento, resultado e reprodução

O [documento v1](../bench/results/2026-10-17-hip-kv-q8-t4-v2/provenance/AMD_HIP_KV_Q8.v1.md)
foi arquivado por bytes antes de atualizar a especificação viva. Os testes e relatórios históricos,
incluindo o FAIL v1 de chat-130K e as revisões T5, permanecem intactos.

A [reanálise dos quinze casos](../bench/results/2026-10-17-hip-kv-q8-t4-v2/README.md) usa os dados
existentes após a escolha da regra: **é reanálise posterior, não uma campanha confirmatória
independente**. Nenhuma inferência GPU nova. Uma futura campanha independente pode usar
esta versão identificada e fixada antes dos novos outputs.

Na raiz:

```powershell
.venv/Scripts/python.exe -B tools/hip/test_kv_q8_t4_policy.py
.venv/Scripts/python.exe bench/results/2026-10-17-hip-kv-q8-t4-v2/reanalyze.py run
```

O primeiro comando verifica o avaliador, incluindo os dois casos que mudam de decisão e
as recusas de métricas inválidas. O segundo recalcula as médias/EP dos 7680 pares preservados,
aplica v2, confere o cálculo por uma via Decimal independente e escreve apenas na pasta nova.
Não executar `prepare` novamente: esse passo conserva o primeiro congelamento antes da edição.
