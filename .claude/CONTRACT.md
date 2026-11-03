# PROTOCOLO DE COORDENAÇÃO — v1.0

Documento normativo. Todo agente deste sistema obedece a ele. Em conflito entre este
arquivo e o system prompt de um agente, **este arquivo vence**, exceto nas seções
marcadas `[LOCAL]` dentro do agente.

---

## 1. Envelope de saída (obrigatório)

Todo agente encerra sua execução com **exatamente um** bloco abaixo, como última coisa
da resposta. Sem prosa depois dele. O dispatcher faz parse por regex do bloco ```json.

```json
{
  "agent": "coder",
  "task_id": "T-042",
  "status": "DONE",
  "summary": "1-3 frases factuais. Sem adjetivo, sem autoelogio.",
  "artifacts": [
    { "path": "src/features/meals/useMealLog.ts", "action": "modified", "lines": 47 }
  ],
  "findings": [],
  "deferred": [
    { "what": "extrair hook duplicado em useHydration", "why": "fora do escopo de T-042" }
  ],
  "handoff": { "next": "tester", "reason": "implementação completa, sem testes" },
  "budget": { "turns_used": 12, "files_touched": 3, "loop_iteration": 1 },
  "confidence": "high"
}
```

### Campos
| Campo | Regra |
|---|---|
| `status` | `DONE` \| `BLOCKED` \| `REJECTED` \| `ESCALATE` \| `FAIL` |
| `findings` | Só preenchido por reviewer / security / tester. Ver §3. |
| `deferred` | Tudo que o agente **viu e decidiu não fazer**. Ver §4. |
| `handoff.next` | Nome do próximo agente ou `orchestrator`. Nunca vazio. |
| `confidence` | `high` \| `medium` \| `low`. `low` obriga `handoff.next = orchestrator`. |

### Semântica de status
- **DONE** — contrato de saída satisfeito integralmente.
- **BLOCKED** — dependência externa ausente (credencial, serviço fora, decisão do humano).
- **REJECTED** — o *input* violou o contrato de entrada (§2). Não tente consertar o input.
- **ESCALATE** — limite de loop, orçamento ou ambiguidade irredutível.
- **FAIL** — tentou, não conseguiu, e o motivo é técnico. Exige `findings` explicando.

---

## 2. Contrato de entrada — regra do REJECT rápido

Cada agente declara no seu prompt os campos que **precisa** receber. Se faltar qualquer um:

> **PARE IMEDIATAMENTE. Emita `status: "REJECTED"` listando os campos ausentes.**

Proibido: reconstruir contexto faltante lendo o repositório inteiro, inferir requisito
não escrito, ou "fazer o melhor possível com o que tem". Essa é a fonte nº 1 de overwork
e de trabalho descartado. Um REJECT custa 2 turnos; uma inferência errada custa um ciclo
completo de 6 agentes.

---

## 3. Severidade — taxonomia única

| Nível | Definição operacional | Destino |
|---|---|---|
| `BLOCKER` | Quebra em produção, perda/corrupção de dado, vazamento de credencial ou PII, bypass de authz, build/CI vermelho. | Fixer, imediato |
| `MAJOR` | Bug em caminho realista não-crítico, race condition sob carga, ausência de tratamento de erro em I/O, violação do design system canônico, regressão de performance mensurável. | Fixer, mesmo ciclo |
| `MINOR` | Legibilidade, duplicação, nome ruim, teste frágil, tipo largo demais. | `deferred[]` → backlog |
| `NIT` | Estilo, formatação, preferência pessoal. | **Não reportar.** Linter resolve. |

Regras duras:
- Um finding sem `evidence` (arquivo:linha + trecho ou log) **não existe**. Descarte-o.
- Só `BLOCKER` e `MAJOR` chegam ao Fixer. `MINOR` nunca dispara retrabalho no ciclo atual.
- Proibido inflar severidade para garantir atenção. Reincidência é falha do agente.

---

## 4. Regra de não-proatividade

Nenhum agente executa trabalho não solicitado. Melhoria adjacente que você identificou:
vai para `deferred[]` com justificativa de uma linha. Nunca para o diff.

Isso inclui, e não se limita a: renomear variável fora do escopo, atualizar dependência,
formatar arquivo vizinho, "já que estou aqui" de qualquer natureza, criar README não pedido,
adicionar log de debug permanente.

---

## 5. Loop breaker (anti-overwork estrutural)

```
tester → (fail) → fixer → tester → (fail) → fixer → tester → (fail) → ESCALATE
```

- `budget.loop_iteration` é incrementado pelo Fixer e propagado pelo Tester.
- **Iteração 3 é a última.** Na falha da 3ª, Tester emite `ESCALATE` para o Orchestrator
  com o histórico das 3 tentativas. Proibido iniciar uma 4ª.
- Se o Fixer receber o mesmo `finding.id` duas vezes, o diagnóstico está errado, não a
  correção: emitir `ESCALATE` imediatamente em vez de tentar variação da mesma correção.

---

## 6. Orçamento por agente

Cada agente tem teto de `maxTurns` (frontmatter) e teto de arquivos tocados (prompt).
Ao atingir 80% de qualquer teto: **pare, entregue o parcial, emita `ESCALATE`**.
Entregar 70% com fronteira honesta é superior a estourar o contexto e devolver ruído.

---

## 7. Fronteiras de escrita

| Agente | Pode escrever em |
|---|---|
| orchestrator | `.claude/state/**`, `TASKS.json`, `AGENT_LOG.md` |
| architect | `docs/adr/**`, `docs/specs/**` |
| researcher | nada (sem `Write`; devolve o relatório na resposta e o orchestrator grava em `docs/research/**`) |
| coder | apenas os paths listados em `spec.files[]` |
| tester | `**/*.test.*`, `**/*.spec.*`, `tests/**`, `e2e/**`, `src/eval/**` (harness de métricas, override do projeto) |
| reviewer | nada |
| security | nada |
| fixer | apenas paths presentes em `findings[].file` |
| devops | `infra/**`, `iac/**`, `cdk/**`, `terraform/**`, `.github/workflows/**`, `Dockerfile*`, `scripts/ops/**`, `docs/runbooks/**`, config de deploy |

Escrever fora da própria faixa é violação de contrato: aborte e emita `REJECTED`.

---

## 8. Ambiente (herdado do setup do Felipe)

Override do projeto astro-triage (Python). Substitui o §8 global, que é de React Native.

- Windows / PowerShell: use `Select-String`, não `grep`. `rg` se disponível. Sintaxe `$env:VAR`.
- Python 3.12, `venv` + `pip`. Sempre o caminho completo `.venv\Scripts\python.exe`.
  O PATH pode não atualizar em terminais já abertos.
- Git é case-insensitive no Windows e case-sensitive no CI Linux — verifique casing de
  imports e nomes de arquivo antes de declarar DONE.
- Dados brutos nunca entram no git (`data/`, `logs/`, `*.parquet`, `*.duckdb`). Só scripts,
  queries e manifestos.
- Chamadas de LLM só à API da NVIDIA (ADR 0005), com `NVIDIA_API_KEY` do ambiente ou do `.env`
  (nunca abrir, ler ou imprimir a chave). Qualquer execução além do piloto de 100 objetos do val
  exige aprovação humana explícita de `max_calls` e `max_tokens_total`. O guarda de cota é checado
  antes de cada chamada. Nada de AWS, Bedrock ou provedor pago sem uma nova ADR.
- Sem commit sem aprovação humana. Conventional commits. Sem trailer de coautoria.
- Código, identificadores e ADRs em inglês. README em inglês e português.

---

## 9. Protocolo de consulta ao grafo (graphify)

Aplica-se aos agentes que possuem acesso a `graphify`. Precondição: existe
`graphify-out/graph.json` no repositório.

### Precedência para entender o repositório

1. `graphify query "<pergunta>"` — subgrafo escopado para a pergunta
2. `graphify path "A" "B"` — como duas coisas se conectam
3. `graphify explain "<símbolo>"` — vizinhança e grau de um nó
4. `Read` / `Grep` / `Glob` — **só depois**, e só nos arquivos que o grafo apontou

Grepar o repositório inteiro antes de consultar o grafo é desperdício de orçamento.

### Gate de frescor (obrigatório antes da primeira consulta)

O grafo está **suspeito** se qualquer condição for verdadeira:

- `graphify-out/graph.json` não existe
- o mtime de `graph.json` é anterior ao último commit
- uma query retorna nó cujo `Source: arquivo:linha` não corresponde ao arquivo real

Nesse caso: **não confie no grafo**. Caia para `Read`/`Grep`, siga a tarefa normalmente e
registre em `deferred[]` que o grafo precisa de `graphify update .`. Você **nunca**
reconstrói o grafo por conta própria — rebuild é do humano ou do `devops`.

Grafo ausente **não** é motivo de `REJECTED`. É degradação, não bloqueio.

### O que o grafo responde — e o que não responde

| Responde | Não responde |
|---|---|
| onde algo está, o que chama o quê | se o código está correto |
| o que quebra se eu mudar X | o que acontece em runtime |
| qual o precedente deste padrão no repo | valores concretos, config efetiva |
| que arquivos uma feature toca | se um teste passa |

Para a coluna da direita, leia o arquivo. Sempre.

### EXTRACTED vs INFERRED

Toda aresta vem marcada `EXTRACTED` (explícita na fonte) ou `INFERRED` (resolvida pelo
graphify). Decisão de peso — ADR, veredito de review, escopo de spec, achado de segurança —
só se apoia em `EXTRACTED`. `INFERRED` é hipótese: confirme lendo o arquivo antes de
afirmar. Citar `INFERRED` como fato é falha de contrato.

### Teto

Máx. 3 consultas ao grafo por tarefa. O grafo é atalho para chegar ao arquivo certo, não
substituto de ler o arquivo certo.
