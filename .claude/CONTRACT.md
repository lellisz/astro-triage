# Contrato de agentes v2.0

Normativo para os nove agentes do pipeline. Em conflito com o prompt de um agente, vale este
contrato. O perfil local (§8) vence os padrões gerais das seções 1 a 7. A fonte global deste
texto é a skill `contrato-agentes`; a v1.0 deste arquivo está no histórico do git.

## 1. Entrada

Quem delega envia `goal` (uma frase verificável) e os campos de intenção do agente.
Campos de contexto o agente descobre sozinho.

| Agente | Intenção (só quem pede sabe) | Contexto (descubra) |
|---|---|---|
| orchestrator | pedido em linguagem natural | todo o resto |
| architect | `goal` | `constraints`, estado do repositório; `non_goals` (se ausente, declare os seus) |
| researcher | `question`, `decision_context` | `versions` (lockfile, manifest) |
| coder | `goal`, `acceptance[]`, `files[]` | `constraints`, comando `verify` |
| tester | `acceptance[]` | arquivos alterados (git), comando de teste, `loop_iteration` (padrão 0) |
| reviewer | `goal` do diff; `acceptance[]` se existir | o diff (padrão: alterações não commitadas; ou o range informado) |
| security | `goal` do diff | o diff; `trust_boundaries` e `data_classes` (perfil do projeto; sem perfil, derive do código) |
| fixer | `findings[]` BLOCKER ou MAJOR completos, `loop_iteration` | comando `verify` |
| devops | `goal`, `environment`, `acceptance[]`; `blast_radius` se produção ou irreversível | região, conta, orçamento, estado atual |

- Intenção ausente ou contraditória: `REJECTED`, listando o que falta. Não adivinhe intenção.
- Contexto ausente: procure no perfil do projeto, nos manifests, no git e na configuração de
  CI, com no máximo 5 chamadas de leitura. Achou: siga e registre em `assumptions`.
  Não achou: `BLOCKED` com `needs`.
- `acceptance[]` é binário: um terceiro sem contexto responde sim ou não.

## 2. Saída: envelope

A resposta termina com exatamente um bloco, sem texto depois:

```json
{
  "contract": "2.0",
  "agent": "coder",
  "task_id": "T-042",
  "status": "DONE",
  "verdict": null,
  "summary": "Uma a três frases factuais.",
  "artifacts": [{ "path": "src/x.py", "action": "modified" }],
  "verification": [
    { "command": "python -m pytest -q", "exit_code": 0, "output_tail": "12 passed in 0.8s" }
  ],
  "findings": [],
  "assumptions": [],
  "deferred": [{ "what": "duplicação em y.py", "why": "fora do escopo" }],
  "needs": null,
  "handoff": { "next": "tester", "reason": "implementado, sem testes novos" },
  "budget": { "files_touched": 1, "loop_iteration": 0 },
  "confidence": "high"
}
```

| Campo | Regra |
|---|---|
| `status` | Como a execução terminou. Ver §3. |
| `verdict` | `PASS`, `FAIL` ou `null`. Só tester, reviewer e security julgam. `FAIL` se houver finding BLOCKER ou MAJOR, ou critério de aceite não demonstrado. |
| `verification` | Comandos realmente executados, com código de saída e as últimas linhas literais. Obrigatório para `DONE` de coder, tester, fixer e devops. |
| `acceptance_map` | Só o tester: uma linha por critério de aceite, com `criterion`, `test`, `result` (`PASS` ou `FAIL`) e `command`. |
| `findings` | Ver §4. |
| `assumptions` | O que você descobriu ou supôs por falta de contexto. |
| `deferred` | O que você viu e decidiu não fazer, com o motivo. |
| `needs` | Obrigatório em `BLOCKED`: `{ "type": "approval", "detail": "..." }`. Tipos: `approval`, `decision`, `input`. |
| `handoff.next` | Sugestão de próximo passo. Quem roteia é quem chamou. |
| `budget.loop_iteration` | Número de correções já aplicadas neste ciclo. Começa em 0. |
| `confidence` | `high`, `medium` ou `low`. Critério que só um humano confirma: no máximo `medium`. |

`task_id` é `null` quando quem delegou não informou.

## 3. Status

- `DONE`: o papel foi cumprido e a evidência está no envelope. Tester que encontrou falha
  devolve `DONE` com `verdict: FAIL`.
- `BLOCKED`: falta algo externo: aprovação, decisão, credencial, contexto que não deu para
  descobrir, arquivo fora da sua faixa de escrita. Diga o que é em `needs`.
- `REJECTED`: intenção ausente ou contraditória.
- `ESCALATE`: limite de ciclo, de arquivos, de diff ou de escopo. Entregue o parcial coerente.
- `FAIL`: tentou e não conseguiu por motivo técnico. Mostre a falha em `verification`.

Sem execução real não existe `DONE`. Código escrito sem comando rodado é `FAIL` ou `BLOCKED`.

## 4. Findings

```json
{ "id": "REV-01", "severity": "MAJOR", "file": "src/x.py", "line": 88,
  "claim": "o que está errado", "evidence": "trecho ou log literal",
  "fix_hint": "direção, não o código pronto" }
```

- Prefixo do `id` por origem: `REV-` reviewer, `SEC-` security, `TST-` tester. O `id` não
  muda entre iterações.
- Sem `evidence` o finding não existe.
- Campos extras por papel são permitidos. Security: `vector`, `precondition`, `impact`, `cwe`.
  Tester: `repro`, `expected`, `actual`.

| Severidade | Definição | Destino |
|---|---|---|
| `BLOCKER` | Quebra em produção, perda ou corrupção de dado, vazamento de credencial ou dado pessoal, bypass de autorização, build ou CI vermelho. | fixer, imediato |
| `MAJOR` | Bug em caminho realista, condição de corrida, I/O sem tratamento de erro, violação de regra declarada no perfil do projeto, regressão de performance mensurável. | fixer, mesmo ciclo |
| `MINOR` | Legibilidade, duplicação, nome ruim, teste frágil. | registrar, sem retrabalho |

Estilo e formatação não são finding. Não infle severidade.

## 5. Ciclo de correção

`tester (FAIL) → fixer → tester (FAIL) → fixer → tester (FAIL) → ESCALATE`

- No máximo duas correções por ciclo. O fixer devolve `ESCALATE` quando recebe
  `loop_iteration` igual a 2.
- O mesmo `id` de volta depois de corrigido: o diagnóstico está errado. `ESCALATE` imediato.
- Só BLOCKER e MAJOR chegam ao fixer.
- Fechamento: todo finding corrigido fecha com prova. Primeira opção: teste de regressão do
  tester que falharia antes da correção. Quando não há teste possível: quem emitiu o finding
  reexamina só os trechos alterados. Sem prova o finding continua aberto.

## 6. Limites

- Trabalho não pedido vai para `deferred`, nunca para o diff.
- Cada agente escreve só na faixa do próprio prompt. O perfil do projeto pode redefinir a
  faixa. Precisou sair dela: `BLOCKED`.
- Os tetos de arquivos e de diff estão no prompt de cada agente. Estourou: `ESCALATE` com o
  parcial.
- Quem delega e recebe uma resposta sem envelope trata o agente como interrompido: retoma
  uma vez; se repetir, assume `ESCALATE`.

## 7. Definição de pronto

Quem coordena só reporta "concluído" quando:

1. cada critério de `acceptance` tem linha `PASS` no `acceptance_map` do tester, com o
   comando e a saída;
2. não há finding BLOCKER ou MAJOR aberto;
3. o `verify` do projeto saiu com código 0 depois da última alteração;
4. os gates do tier rodaram: reviewer; security quando houve gatilho.

Em trabalho sem tester (T0 e T1) valem os itens 2 e 3. Faltando qualquer item, o relatório
diz "não validado" e lista o que falta.

## 8. Perfil local: astro-triage (Python)

Redefine ou acrescenta ao que vale acima, só para este repositório.

**Faixa de escrita**
- `tester`: além de `tests/**`, escreve em `src/eval/**` (harness de métricas).
- `researcher` não escreve; o orchestrator grava o relatório em `docs/research/**`.

**Ambiente**
- Windows / PowerShell: use `Select-String`, não `grep`. `rg` se disponível. Sintaxe `$env:VAR`.
- Python 3.12, `venv` + `pip`. Sempre o caminho completo `.venv\Scripts\python.exe`.
  O PATH pode não atualizar em terminais já abertos.
- Git é case-insensitive no Windows e case-sensitive no CI Linux: verifique o casing de
  imports e nomes de arquivo antes de declarar `DONE`.
- Dados brutos nunca entram no git (`data/`, `logs/`, `*.parquet`, `*.duckdb`). Só scripts,
  queries e manifestos.
- Chamadas de LLM só à API da NVIDIA (ADR 0005), com `NVIDIA_API_KEY` do ambiente ou do `.env`
  (nunca abrir, ler ou imprimir a chave). Qualquer execução além do piloto de 100 objetos do
  val exige aprovação humana explícita de `max_calls` e `max_tokens_total`. O guarda de cota é
  checado antes de cada chamada. Nada de AWS, Bedrock ou provedor pago sem uma nova ADR.
- Kaggle: só recursos privados `astro-triage-*`, só código, sem dado bruto. Toda escrita no
  Kaggle e todo download de saída exigem o sim explícito do dono na conversa.
- Sem commit sem aprovação humana. Conventional commits. Trailer `Co-Authored-By` permitido
  desde 2026-11-03.
- Código, identificadores e ADRs em inglês. README em inglês e português.

**Grafo (graphify)**
Siga a skill `protocolo-grafo`: confira o frescor antes de confiar, decisão de peso só sobre
aresta `EXTRACTED`, máximo de 3 consultas por tarefa, e nunca reconstrua o grafo por conta
própria (registre em `deferred` que precisa de `graphify update .`).
