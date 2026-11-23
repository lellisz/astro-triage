# ADR 0007: Reexecução determinística da avaliação de teste do deep learning após a perda de `data/` (2026-11-17)

- Status: **Accepted** (2026-11-18, Q0 aprovada pelo dono na conversa depois do `reviewer` sem
  BLOCKER/MAJOR). Aceitar a ADR autoriza só os passos seguintes da seção 12, na ordem; **não autoriza abrir
  o teste**: isso só acontece com A1 e A2, depois das checagens P0, E e F.
- Data: 2026-11-17
- Tarefa: não informada (rascunho pedido pelo orchestrator)
- Relacionadas: ADR 0006 (protocolo pré-registrado; seção 5, congelamento de validação; seção 14 e nota da
  seção 17, abertura única do teste e marcador `test_opened.json`; Amendment B, seção 18, registro do A6 de
  2026-11-16), ADR 0003 (split por pixel, piso de reporte, bootstrap), ADR 0005 (Layer 2, **continua
  fechada**: Amendment 2, seção 18), ADR 0001 (escrita temp+rename, escritor único).
- Fora de escopo: retreinar qualquer modelo (ver a nota do dono de 2026-11-18 na seção 2, que não autoriza
  nada por si); mudar qualquer constante, limiar, prior, hiperparâmetro, semente, regra de decisão ou par da
  ADR 0006; o kernel v7 (ADR 0006, 18.3: não usado); Layer 2 / LLM; AWS ou qualquer provedor pago; o
  dashboard (ele só relê os arquivos regenerados); editar a ADR 0006 (uma nota datada nela é tarefa
  separada, seção 15).
- Idioma: português do Brasil, por decisão explícita do dono em 2026-11-18, como exceção ao
  `.claude/CONTRACT.md` (seção 8: ADRs em inglês).
- Nenhuma métrica de teste é citada aqui como valor. A referência numérica da avaliação original é a tabela
  da ADR 0006, seção 18.4, e continua **PROVISIONAL** (`CLAUDE.md`, Status). Os hashes e caminhos citados
  vêm de `docs/results/MANIFEST.md` (gerado 2026-11-16T20:51:20Z, commit `d150490`) ou foram lidos dos
  arquivos em 2026-11-17, com a fonte ao lado.

Rótulos usados nesta ADR: D1 a D3 são as decisões de design das seções 3 a 5; Q0 a Q3 são as perguntas ao
dono na seção 12; A1 a A3 são as aprovações do dono na seção 12.

## 1. Contexto

### 1.1 O acidente

Em 2026-11-17, um `git worktree remove --force` atravessou uma junction criada dentro de uma worktree de
agente e esvaziou o conteúdo real de `data/` do repositório principal. A causa e a regra nova (nunca ligar
dados por junction ou symlink dentro de uma worktree; preferir variável de ambiente ou parâmetro apontando
para o caminho original, somente leitura) estão em `~/.claude/CLAUDE.md`, seção "Worktree por agente".
Nenhum dado entrou no git (regra do repositório), então o git não tem cópia.

Perdidos e ainda não regenerados, em `data/interim/full_v0/dl/`:

| arquivo | sha256 original (`MANIFEST.md`) | linha |
|---|---|---|
| `eval/results_dl.json` | `037e684d...` | `MANIFEST.md:11` |
| `eval/results_dl.md` | `c86a887e...` | `MANIFEST.md:10` |
| `eval/risk_coverage_dl.csv` | `42f3dea8...` | `MANIFEST.md:12` |
| `val_freeze.json` | `ca360bf6...` | `MANIFEST.md:13` |
| `test_opened.json` | `fe77a2e9...` | `MANIFEST.md:14` |

### 1.2 O que foi recuperado (fatos verificados por quem encomendou esta ADR)

- Base LightGBM em `lgbm_seeds/`: as 10 predições `predictions_lgbm_{full,noind}_s{0..4}_v0.parquet` e
  `delta.json`, conferidas por SHA-256 contra `MANIFEST.md:17,49-58`.
- Predições do DL, 5 sementes x 4 variantes (`mlp_full`, `mlp_noind`, `ftt_full`, `ftt_noind`), 20 arquivos,
  vindas das versões v2 a v6 do notebook `<conta-kaggle>/astro-triage-dl`, conferidas por SHA-256 contra
  `MANIFEST.md:29-48`. As linhas de validação dessas 20 predições são idênticas às originais (consequência
  direta da igualdade de hash).
- Os 10 ensembles `ens_{mlp,ftt}_full_s{0..4}` foram regenerados localmente por `scripts/train_dl.py
  --ensemble`. **Os bytes não batem com `MANIFEST.md:19-28`.**
- `val_freeze.json` ainda não foi regenerado: `train_dl.py --freeze` exige o texto de aprovação do dono.

### 1.3 Fatos lidos no repositório para esta ADR (2026-11-17)

- F1. Por que os bytes dos ensembles não podem bater: `build_ensemble` grava nos metadados do parquet
  `created_utc = datetime.now(UTC)` e `git_sha = git rev-parse HEAD` (`src/astro_triage/models/ensemble.py:138-139`).
  Qualquer reconstrução em outro instante ou outro commit muda esses dois campos e, portanto, o hash do
  arquivo, mesmo com valores idênticos. A diferença de bytes é **esperada**; ela não prova divergência de
  valores, mas também não prova igualdade.
- F2. O código não mudou: `git diff --stat 3444c99 HEAD` sobre `models/ensemble.py`, `models/lgbm.py`,
  `models/dl_train.py`, `evaluation/dl_val.py` e `scripts/train_dl.py`, e `git diff --stat 0d0d918 HEAD`
  sobre `src/astro_triage/evaluation/`, `src/eval/`, `scripts/evaluate_dl.py`, `src/astro_triage/split.py` e
  `src/astro_triage/dataset/`, saíram vazios. `0d0d918` é o `git_sha` registrado no A6 (ADR 0006, 18.1).
  Os ensembles originais foram obtidos entre 19:08:41Z e 19:09:17Z de 2026-11-16 (`MANIFEST.md:19-28`),
  depois de `3444c99`/`326d67a` (16:07:50-03:00) e antes de `0d0d918` (16:12:17-03:00).
- F3. O `val_freeze.json` original não pode ser reproduzido byte a byte: ele contém o texto de aprovação
  (`dl_val.build_freeze`, campo `approval`) e o `prediction_sha256` de todos os 40 sistemas planejados,
  inclusive dos 10 ensembles (`src/astro_triage/evaluation/dl_val.py:389,427-431`). O novo freeze terá
  outro sha256, e o novo `results_dl.json` e o novo `test_opened.json` vão referenciar esse sha novo.
- F4. O arquivo atual `dl/train_report_dl.json` **já foi reescrito** pela regeneração: a entrada
  `ens_mlp_full_s0` registra `prediction_sha256` `af7ce925...`, diferente do original `c6d98fae...`
  (`MANIFEST.md:24`). Portanto ele não é o original (`1d1d6efa...`, `MANIFEST.md:15`) e seus campos de
  ensemble (`weight`, `temperature`, `val_purity`) não servem de referência original. As 20 entradas
  não-ensemble podem ser originais; isso precisa ser conferido (seção 4, P0.4).
- F5. O `labels.parquet` tem referência original independente: `labels_sha256` `70a69da5...` está em
  `lgbm_seeds/delta.json` (sha conferido) e em `full_v0_labels.manifest.json`.
- F6. As únicas referências originais de valor para a saída da avaliação são: a tabela e a saída do
  comando C3 da ADR 0006, seção 18.4 (as quatro células headline: `d`, `ci`, `delta`, `blocks`,
  `reportable`, veredito de classe e de par; e `n_test`, `n_test_pixels`, `n_boot`, `seed`). As células
  descritivas (ensembles, frentes 2 a 4) não têm cópia fora de `data/` (`Grep` por `ens_mlp_full` e pelos
  shas `037e684d`/`ca360bf6` fora de `data/`, `src/`, `tests/`: só `MANIFEST.md` e a ADR 0006).
- F7. A guarda de abertura única é um arquivo local: `dl_eval.open_test_set` recusa se
  `dl/test_opened.json` existe (`src/astro_triage/evaluation/dl_eval.py:171-172`). Com `data/` apagado, a
  guarda sumiu e o código abriria o teste de novo sem reclamar. O que impede a reabertura agora é a regra do
  repositório (reabrir exige ADR nova), não o código.
- F8. Os pares headline (H1 `mlp_full-lgbm_full`, H2 `ftt_full-lgbm_full`) não usam ensembles (ADR 0006,
  seção 16, item 5: ensembles são descritivos e sem veredito). Os veredictos headline dependem só de
  arquivos já conferidos por hash.
- F9. Os limiares `val_purity` de cada fit DL no relatório de treino vêm de `dl_val.val_purity_summary`
  (`src/astro_triage/models/dl_session.py:233`), que usa a mesma função `_p2_val` do `build_freeze`.

### 1.4 Por que reavaliar

A ADR 0006 e o dashboard citam `results_dl.{json,md}`, `risk_coverage_dl.csv`, `val_freeze.json` e
`test_opened.json` como evidência. Sem eles, o resultado do A6 existe só como texto na ADR 0006, seção
18.4, sem o artefato que o comprova, e o dashboard perde a fonte da aba de comparação.

## 2. Opções

| id | opção | custo real | consequência |
|---|---|---|---|
| O1 | **Reexecução determinística** no código atual (idêntico ao de `0d0d918`, F2), com as entradas conferidas por hash, novo freeze e uma única nova execução de `evaluate_dl.py` | uma sessão local de CPU (bootstrap B = 1000); duas aprovações do dono; uma checagem pós-execução | Os artefatos voltam a existir. É uma segunda leitura do teste, mas sem nenhum grau de liberdade: tudo o que poderia mudar o resultado está fixado por hash ou por código idêntico. Os hashes de freeze, resultados e marcador mudam (F1, F3) |
| O2 | **Retreinar no Kaggle** (rejeitada como caminho padrão; ver a nota do dono abaixo) | GPU semanal desconhecida (ADR 0006, 9.6); para o FT-Transformer, horas por semente (ADR 0006, 18.2); cada escrita no Kaggle exige um sim do dono | O GPU não é determinístico bit a bit (ADR 0006, seção 7, R6): os modelos novos não reproduzem as predições originais, então o resultado seria **outro experimento**, treinado e avaliado depois de o teste ter sido visto (pós-teste, ADR 0006, 4.4). Desnecessário: as 20 predições originais já foram recuperadas com hash igual |
| O3 | **Não reavaliar** (rejeitada) | zero computação | O resultado do A6 fica sem artefato: a seção 18.4 da ADR 0006 cita comandos (C3, C4) sobre arquivos que não existem; o critério 5 da seção 16 da ADR 0006 deixa de ser verificável; a aba de comparação do dashboard fica sem fonte; a ADR 0006 não tem como sair de Proposed com evidência. O teste não é reaberto |
| O4 | **Reavaliar só os pares headline** (rejeitada) | exige mudar `evaluate_dl.py`/`dl_run.py` para pular sistemas | Muda o código do avaliador depois de o teste ter sido visto, e o relatório deixa de ser o mesmo relatório pré-registrado. Contraria a regra "reproduzir, não ajustar" |

Nota do dono (2026-11-18, na conversa): "se for necessário, o DL pode ser treinado de novo no Kaggle". Isso se
liga à opção O2 e **não autoriza nada por si**:

- (i) É uma opção condicional: só entra em consideração se uma das checagens de equivalência (P0 ou E,
  seção 4) falhar e mostrar que a reexecução determinística (O1) não é possível.
- (ii) Reabre a questão da exposição única do teste e do freeze de validação da ADR 0006. Um caminho com
  retreino exigiria ADR nova ou emenda própria, com novo protocolo e novo freeze, **antes** de qualquer
  execução. Esta ADR não define protocolo, custo nem prazo para esse retreino.
- (iii) Vale a regra do projeto para o Kaggle: só recursos privados `astro-triage-*`, só código sem dados
  brutos, e cada escrita no Kaggle ou download de saída precisa de sim explícito do dono na conversa.

## 3. Decisão (D1): O1, reexecução determinística, sob condições de equivalência verificáveis

1. A reexecução **reproduz** a avaliação de 2026-11-16 (A6) e nada mais. Não se permite, antes ou depois de
   ver o teste: mudar qualquer limiar, constante de `dl_val.current_constants()`, prior, nível de
   completude, par, semente, `n_boot`, hiperparâmetro, arquitetura, peso de ensemble, temperatura,
   conjunto de sistemas, regra de decisão ou texto de veredito; retreinar ou substituir qualquer predição;
   usar o kernel v7; rodar o avaliador com argumentos diferentes dos padrão, exceto `--approval`.
2. O código usado é o do commit em que a execução acontece, desde que a checagem P0.1 (seção 4) mostre
   diff vazio em relação a `0d0d918` e `3444c99` nos caminhos de F2. Diff não vazio: parar.
3. A execução roda na **árvore principal**, não em worktree (o avaliador exige `--outdir` dentro de
   `data/` do próprio repositório, `scripts/evaluate_dl.py:45`; nada de junction, regra nova de
   2026-11-17).
4. Uma única execução de `evaluate_dl.py` está autorizada por esta ADR. A fronteira da abertura **não é
   vista no disco**: a evidência é o stderr literal e o código de saída (o código é 1 em toda falha, então
   só o stderr distingue os casos). Estado do disco por ponto de falha, pelo código
   (`scripts/evaluate_dl.py:44-71`, `src/astro_triage/evaluation/dl_eval.py:165-194`):
   - (i) Recusa **antes da leitura**: nada de teste foi lido e nada foi gravado. São só estas mensagens
     (todas com o prefixo `dl evaluation refused: `), que não podem ocorrer depois da leitura:
     `--outdir and the dirnames must resolve inside the repo's data/`; `outputs already exist, nothing is
     overwritten` (só vale aqui se os arquivos de saída já existiam em disco antes da invocação,
     conferido por P0.5; o mesmo texto sai da segunda checagem, `evaluate_dl.py:55`, feita depois de o
     teste ser lido); as de `dl_desc.load_costs` (`cannot read costs from ...`, `... fits is not an
     object`); `test opening needs the owner's explicit approval text (A6)`; `test already opened`; e
     `test set was already opened in this process; it is opened once`.
     **Não** estão nesta lista as recusas de `verify_freeze` e `load_freeze` (`... is a pilot directory
     ...`, `val_freeze.json missing ...`, `cannot read ...`, `freeze constants differ ...`, `... planned
     files missing ...`, `... sha256 differs from the freeze ...`, `freeze thresholds ...` e as demais
     dessa faixa): `load_dl_systems` chama `verify_freeze` de novo (`dl_eval.py:217`) depois de o teste
     ser lido, e o stderr não distingue as duas chamadas. Também são pós-leitura, e ficam fora da
     lista, `cannot read <nome do arquivo>: ...` (`dl_eval.py:202`, `_read_proba`) e `freeze has no
     train_class_fractions for ...` (`dl_eval.py:231`). Qualquer uma dessas, em execução, conta como
     abertura; o custo é baixo, porque a checagem F acabou de verificar o mesmo freeze.
   - (ii) Falha **dentro de `load_test_set`** (`dl_eval.py:177`, depois de `_TEST_OPENED` marcado na
     linha 176): ocorre antes de `open_test_set` retornar, mas pode já ter lido rótulos de teste. O disco
     fica sem resultados e sem marcador, parecendo "nunca aberto".
   - (iii) Falha **entre o retorno de `open_test_set` e a gravação dos resultados** (inclui a segunda
     checagem `_free`): nada foi gravado, ou os arquivos parciais são apagados
     (`evaluate_dl.py:57-63`); o disco fica quase idêntico a "nunca aberto" (pode restar o diretório
     `eval/` vazio, criado em `evaluate_dl.py:56`), mas o teste foi lido.
   - (iv) Falha em **`write_marker`** (`evaluate_dl.py:65`): os três arquivos de resultado ficam em disco
     e o `test_opened.json` não existe.

   Regra de fronteira (conservadora): **só as recusas do item (i) não contam como abertura.** Qualquer
   outro erro, traceback ou falha a partir da verificação do freeze conta como abertura já ocorrida, mesmo
   que o prefixo seja `dl evaluation refused: ` (as recusas de `load_test_set` e `load_dl_systems`
   usam o mesmo prefixo) e mesmo que o disco pareça "nunca aberto". Dúvida sobre qual caso ocorreu conta
   como abertura. Abertura já ocorrida: não há nova tentativa com mera aprovação; exige **ADR nova**,
   em todos os casos. Um `test_opened.json` ou arquivo de resultado que já exista nessa situação não é
   apagado. Uma recusa do item (i) não consome a execução única: pode ser repetida sob a **mesma**
   aprovação A2, depois de corrigida a causa sem violar o item 1, desde que o stderr literal e o código de
   saída dela tenham sido registrados na seção 13. Toda invocação de `evaluate_dl.py` tem o stderr literal
   e o código de saída registrados na seção 13, inclusive as que falham.
5. Nenhum número produzido entra como final: todas as métricas continuam **PROVISIONAL**.

## 4. Decisão (D2): condições de equivalência (verificáveis, em ordem)

Cada condição é binária. "Igual" significa igualdade exata (hash idêntico, ou arrays de mesmo dtype e mesmo
formato com `np.array_equal`, ou números de ponto flutuante iguais pelo `==` do Python), salvo onde a
tolerância está escrita. A exceção é R1 (e K2): a reprodução da saída headline é verificada na precisão
publicada (quatro casas decimais, o arredondamento do comando C3 do protocolo), não por igualdade exata dos
valores de ponto flutuante subjacentes. Toda checagem roda **sem ler rótulos de teste nem calcular métricas
de teste**, exceto R1 a R3.

### P0. Entradas (antes de qualquer aprovação)

- P0.1 Código: `git diff --stat 0d0d918 <commit da execução> --` sobre `src/astro_triage/evaluation/`,
  `src/eval/`, `src/astro_triage/split.py`, `src/astro_triage/dataset/`, `scripts/evaluate_dl.py`, e
  `git diff --stat 3444c99 <commit da execução> --` sobre `src/astro_triage/models/` e
  `scripts/train_dl.py`: ambos vazios. `git status` sem alteração nesses caminhos.
- P0.2 Hashes: SHA-256 recalculado de cada um dos 30 arquivos não-ensemble (20 DL + 10 LightGBM) e de
  `lgbm_seeds/delta.json` igual ao de `MANIFEST.md` (versão do commit `d150490`).
- P0.3 Rótulos e entradas da Fase 1: SHA-256 de `labels.parquet` igual a `labels_sha256` de
  `lgbm_seeds/delta.json` (F5); SHA-256 de `features.parquet`, `predictions_lgbm_full_v0.parquet` e
  `predictions_lgbm_noind_v0.parquet` igual ao `file_sha256` dos respectivos `*.manifest.json` (lidos por
  `runner.load_test_set`).
- P0.4 Relatório de treino: em `dl/train_report_dl.json`, cada uma das 20 entradas não-ensemble tem
  `prediction_sha256` igual ao do `MANIFEST.md`. Se alguma divergir, essa entrada não serve de referência em
  F.3 e o fato vai para a seção 13.
- P0.5 Estado limpo: não existem `dl/val_freeze.json`, `dl/test_opened.json` nem `dl/eval/`.

### E. Ensembles (bytes não batem; comparar valores)

Os ensembles originais não existem mais, então não há valor original com que comparar arquivo a arquivo
(F4, F6). A equivalência é estabelecida por construção e por determinismo, nesta ordem:

- E.1 Componentes: o `component_sha256` dos metadados de cada `ens_<net>_full_s<k>` regenerado é igual aos
  shas do `MANIFEST.md` de `lgbm_full_s<k>` e de `<net>_full_s<k>` (o `build_freeze` já recusa se divergir
  do arquivo atual, `dl_val.py:399-409`; aqui a referência é o `MANIFEST.md`).
- E.2 Determinismo: uma segunda regeneração dos 10 ensembles, num diretório temporário fora do repositório,
  produz para cada sistema: `source_id` igual; as seis colunas `p_raw_*` e `p_cal_*` iguais (float32,
  `np.array_equal`); `weight` e `temperature` dos metadados iguais como texto (`repr`). O
  `build_ensemble` lê o componente da rede em `dl_dir` e grava o ensemble no mesmo `dl_dir`
  (`ensemble.py:99-103`); por isso o temporário recebe **cópias** dos 10 arquivos `<net>_full_s<k>`, com
  o SHA-256 da cópia conferido contra o `MANIFEST.md` antes do uso. `outdir` e `lgbm_dir` apontam para
  `data/` em leitura. Nenhuma junction, symlink ou hardlink. A checagem lê e compara só colunas de
  predição, nunca rótulos de teste, e não calcula nenhuma métrica de teste.
- E.3 Diferença só em metadados voláteis: comparando o regenerado em `data/` com o da segunda regeneração,
  as únicas chaves de metadados com valor diferente estão contidas em `{created_utc, git_sha}`.
- E.4 (opcional, prova forte de bytes) Reconstrução: reescrever a tabela regenerada trocando só
  `created_utc` por cada segundo de `[obtained_utc - 10 s, obtained_utc]` do `MANIFEST.md` e `git_sha` por
  cada commit candidato (`326d67a...`, `3444c99...`), na mesma versão de `pyarrow` e com `zstd`. Se algum
  candidato der o SHA-256 do `MANIFEST.md`, a igualdade de bytes com o original está provada. Se nenhum
  der, **não** é falha: a versão do escritor ou outro detalhe de serialização pode explicar; E.1 a E.3
  continuam valendo. O dono decide se roda E.4 (seção 12, Q3).

Falha em E.1, E.2 ou E.3: **parar antes do freeze.** Os ensembles regenerados não são considerados
equivalentes. O dono escolhe (sem ver o teste) entre: (a) não reavaliar (O3); ou (b) aceitar os ensembles
regenerados como **novos sistemas descritivos**, declarados assim em `results_dl.md` e nesta ADR, o que
só é aceitável porque os veredictos headline não dependem deles (F8). A escolha (b) exige texto de
aprovação próprio, anterior ao freeze (seção 12, Q1).

### F. Novo freeze (validação apenas)

O novo `val_freeze.json` terá sha256 diferente do original `ca360bf6...` (F3). Equivalência:

- F.1 `labels_sha256` = `70a69da5...` (F5); `delta_sha256` = `a2b1785a...` (`MANIFEST.md:17`).
- F.2 `prediction_sha256` dos 30 sistemas não-ensemble igual ao `MANIFEST.md`; os 10 de ensemble iguais aos
  arquivos que passaram em E.
- F.3 Limiares. Para os 10 sistemas LightGBM, por classe e `C*`, são iguais aos de `delta.json` (o
  `verify_freeze` já confere, `dl_eval.py:150-160`); isso é comparação com referência original. Para os 20
  sistemas DL não-ensemble, F.3 é uma **checagem de consistência, não uma comparação com o original**: o
  `train_report_dl.json` foi reescrito pela regeneração (F4), então os limiares do novo freeze são
  comparados com `val_purity.<classe>.<C*>.threshold` das entradas que passaram em P0.4 (mesma função, F9),
  o que mostra que freeze e relatório concordam, não que algum dos dois seja igual ao original de
  2026-11-16. Quando falta a referência (entrada que falhou em P0.4, ou chave ausente), F **para** (opção
  mais conservadora): F.3 não é considerado aprovado, o freeze é apagado como no parágrafo abaixo e o caso
  é registrado na seção 13; não há referência alternativa. Os limiares de ensemble não têm referência
  original (F4): ficam cobertos por E.
- F.4 `constants` igual a `current_constants()` (o `verify_freeze` já recusa se divergir).
- F.5 `approval` é o texto aprovado pelo dono na seção 12, item A1, palavra por palavra.

Falha em F.1 a F.4, inclusive a falta de referência de F.3: apagar o freeze novo (que ainda não abriu
nada), registrar a falha na seção 13 e não executar a avaliação. Uma nova tentativa de freeze exige
diagnóstico e aprovação nova.

### R. Saída da avaliação (depois da única execução)

- R1 Headline: rodar de novo o comando C3 da ADR 0006, seção 18.4, sobre o novo `results_dl.json`. A
  saída tem de ser **idêntica**, linha por linha, na precisão publicada (quatro casas decimais, o
  arredondamento do C3; não igualdade exata de ponto flutuante), à "Output on 2026-11-16" registrada lá,
  exceto pelos dois campos que mudam por construção: o `git_sha` curto e o prefixo do `freeze_sha256`. A
  comparação inclui `n_test`, `n_test_pixels`, `n_boot`, `seed`, os quatro `d`, `ci`, meia-largura,
  `delta`, `blocks`, `reportable`, veredito de classe, papel e veredito de par.
- R2 Estrutura: o comando C4 da ADR 0006, seção 18.5, sobre o novo arquivo, dá o mesmo total de células e
  zero ausentes, com o mesmo conjunto de pares (papel, tem veredito) descrito lá.
- R3 Descritivo: as células de ensemble e das frentes 2 a 4 não têm referência original (F6). Elas entram
  rotuladas como "regeneradas em 2026-11-17 (ADR 0007), sem comparação com o original", por meio do texto
  de aprovação gravado em `meta.approval` e desta ADR.

Falha em R1 ou R2: a reexecução **não** é uma reprodução. Os arquivos gerados ficam em disco (apagar seria
esconder evidência), mas: nenhum texto do repositório os cita como resultado do A6; o dashboard não é
apontado para eles; a divergência é registrada na seção 13 com a saída literal; não há segunda execução,
nem com parâmetro mudado, nem com o mesmo; qualquer passo seguinte exige outra ADR. O resultado de
referência continua sendo a tabela da ADR 0006, seção 18.4, com a nota de que o artefato que a sustentava
foi perdido e não pôde ser reproduzido.

Qualquer erro, traceback ou falha a partir da verificação do freeze que não seja recusa do item (i) (seção 3,
item 4): mesma regra do parágrafo acima. Nenhuma nova tentativa com mera aprovação; exige **ADR nova**, em
todos os casos. Um `test_opened.json` ou arquivo de resultado que já exista não é apagado, e o fato é
registrado na seção 13.

## 5. Decisão (D3): registro de `test_opened.json` e da proveniência

1. O marcador é escrito pelo código atual, sem mudança de esquema (`schema_version` 1,
   `dl_eval.write_marker`). Mudam, por construção: `opened_at` (data e hora da reexecução, UTC),
   `git_sha` (commit da execução) e `freeze_sha256` (sha do novo freeze). O arquivo perdido de 2026-11-16
   (`opened_at` 2026-11-16T19:22:46+00:00, `git_sha` `0d0d918`, sha256 do arquivo `fe77a2e9...`,
   `MANIFEST.md:14`) continua sendo **a primeira abertura**; esta ADR registra a segunda.
2. A referência ao acidente fica numa cadeia verificável, sem campo novo no marcador:
   `test_opened.json.freeze_sha256` aponta para o novo `val_freeze.json`, cujo `approval` cita o acidente e
   esta ADR (texto A1); `results_dl.json.meta.approval` também cita (texto A2). Alternativa com campo novo
   no marcador: seção 11, gatilho G3.
3. `docs/results/MANIFEST.md` é regenerado por `scripts/make_results_manifest.py` depois da reexecução. A
   versão original continua no git (`git show d150490:docs/results/MANIFEST.md`); a seção 13 desta ADR
   registra lado a lado o sha original e o novo de cada arquivo que mudou.
4. A data da exposição registrada na ADR 0006 (2026-11-16) não muda. Uma nota datada na ADR 0006 apontando
   para esta ADR é tarefa separada (seção 15).

## 6. Riscos

| id | risco | mitigação / medição |
|---|---|---|
| K1 | Ensembles regenerados com valores diferentes dos originais (bytes já diferem, F1) | E.1 a E.3; E.4 opcional; os veredictos headline não dependem deles (F8) |
| K2 | Versão de `numpy`, `pyarrow`, `scipy` ou `lightgbm` no `.venv` mudou desde 2026-11-16 e altera arredondamentos | E.2 detecta não determinismo, não deriva de versão; R1 é a prova final, na precisão publicada (quatro casas decimais, não igualdade exata); registrar `pip freeze` do `.venv` na seção 13 antes do freeze |
| K3 | O número de threads (`--workers`) da execução original não está registrado; métricas descritivas poderiam depender da ordem de soma | usar o padrão do script; R1 cobre o headline; divergência descritiva sem referência fica rotulada (R3) |
| K4 | A guarda de abertura única não existe mais em disco (F7), e em falha o disco pode parecer "nunca aberto" (seção 3, item 4, itens ii e iii): o `test_opened.json` normalmente não existe depois de uma falha | a regra do repositório e esta ADR; P0.5; uma execução autorizada; a fronteira é o stderr literal registrado na seção 13, não o disco: só as recusas do item (i) da seção 3.4 não contam como abertura, qualquer outra falha a partir da verificação do freeze conta e exige ADR nova |
| K5 | Ler o teste de novo induz a tentação de "consertar" algo | seção 3, item 1; R1 compara contra um resultado já publicado, então não há decisão em aberto que o teste possa influenciar |
| K6 | `train_report_dl.json` atual mistura entradas originais e regeneradas (F4) | P0.4 separa as confiáveis; a seção 13 registra quais |
| K7 | Uma nova perda de `data/` antes do registro do MANIFEST novo | regenerar o MANIFEST logo depois de R1/R2; cópia de segurança de `data/` fora do repositório é decisão do dono (fora de escopo) |

## 7. Consequências

- Boas: os artefatos citados pela ADR 0006 voltam a existir; a reprodução do headline a partir de entradas
  conferidas por hash é, ela mesma, evidência de reprodutibilidade do protocolo; o dashboard volta a ter
  fonte.
- Ruins: o teste é lido uma segunda vez; a afirmação "teste aberto uma única vez" passa a ser "teste
  aberto uma vez para decidir (2026-11-16) e uma vez para reproduzir o mesmo cálculo depois da perda de
  dados (data da reexecução)", e todo texto que diga "uma única vez" precisa citar esta ADR. Os hashes de
  freeze, resultados e marcador deixam de bater com os citados na ADR 0006, seção 18.1 e 18.4; quem
  auditar precisa desta ADR para ligar os dois conjuntos. As células descritivas de ensemble passam a ter
  proveniência mais fraca (equivalência por construção, não por comparação).
- Neutras: nenhuma métrica muda de status; tudo continua PROVISIONAL. A ADR 0006 continua Proposed pelos
  motivos dela (18.5).

## 8. Custo de manutenção

Nenhum código novo de produção. Uma checagem de equivalência descartável (seção 15) e um MANIFEST
regenerado. Custo recorrente: zero, exceto que qualquer auditoria futura de 2026-11-16 passa por esta ADR.

## 9. Alternativas rejeitadas (resumo)

- Retreinar no Kaggle (O2): gera outro experimento, não reproduz o original (GPU não determinístico), custa
  quota não medida e cada escrita no Kaggle exige sim do dono; seria um treino pós-teste. A nota do dono de
  2026-11-18 (seção 2) o mantém como opção condicional, sem autorizar nada por si.
- Deixar o teste sem reavaliar (O3): o resultado do A6 fica sem artefato verificável, os comandos C3/C4
  da ADR 0006 não rodam, o dashboard perde a fonte da comparação, e a ADR 0006 não sai de Proposed com
  evidência. Continua sendo o caminho se E ou F falharem e o dono não aceitar (b) da seção 4.
- Reavaliar só o headline (O4): exige mudar o avaliador depois de ver o teste.

## 10. Não-objetivos

Retreinar; mudar a ADR 0006 ou suas constantes; reabrir a Layer 2 / ADR 0005; usar AWS, Bedrock ou
qualquer provedor pago; qualquer escrita no Kaggle ou download de saída (as recuperações já feitas não
fazem parte desta ADR); regenerar o dashboard; tornar qualquer número final; restaurar outros arquivos de
`data/` além dos listados em 1.1; política de backup de `data/`.

## 11. Gatilhos de reversão

- G1. P0.1 com diff não vazio: parar; uma reexecução com código diferente exige outra ADR que justifique a
  mudança.
- G2. Falha em E ou F: seguir o que a seção 4 manda em cada caso; nenhuma nova tentativa sem aprovação
  nova. Falha em R, ou qualquer erro, traceback ou falha a partir da verificação do freeze que não seja
  recusa do item (i) (seção 3, item 4; na dúvida, conta como abertura): nenhuma nova tentativa sem
  **ADR nova**, em todos os casos; mera aprovação não basta.
- G3. O dono quer a referência ao acidente dentro do próprio `test_opened.json`: mudança de esquema para
  `schema_version` 2 com campos `reexecution_of` (`opened_at` e sha256 do marcador original) e `adr`,
  feita **antes** do freeze, com teste, e P0.1 ajustado para aceitar exatamente esse diff em
  `dl_eval.write_marker` e nada mais.
- G4. Aparece uma cópia íntegra dos arquivos originais (sha igual ao `MANIFEST.md`) antes da execução:
  restaurar a cópia e cancelar a reexecução.

## 12. O que o dono precisa aprovar (lista exata)

Cada item é um "sim" explícito na conversa. Nenhum agente executa o passo seguinte sem o anterior. Os
rótulos Q0 a Q3 são perguntas ao dono; A1 a A3 são textos de aprovação; D1 a D3 (decisões de design) estão
nas seções 3 a 5 e não são perguntas.

Sequência completa do que é pedido e quando:

1. **Q0.** Esta ADR: passar de Proposed para Accepted depois do `reviewer`. Nada abaixo começa antes disso.
2. **Q2** e **Q3**, perguntadas logo depois de Q0 e **antes** das checagens do passo 3 e antes do freeze:
   - Q2. Marcador sem campo novo (seção 5, recomendado) ou com campo novo (G3). Se o dono escolher G3, a
     mudança de esquema, o teste e o P0.1 ajustado são feitos **antes** do freeze (passo 5); sem resposta
     a Q2 não há freeze.
   - Q3. Rodar ou não a reconstrução opcional E.4. Deve ser respondida antes de o passo 3 rodar.
3. Rodar P0 e E (E.4 só se Q3 foi sim), somente leitura, sem ler rótulos de teste nem calcular métricas de
   teste.
   - Tudo passa: seguir para A1.
   - Alguma checagem de E falha: parar. **Q1.** Escolher (a) não reavaliar (O3; a sequência termina aqui,
     nada mais é executado) ou (b) aceitar os ensembles regenerados como novos sistemas descritivos, com
     texto de aprovação próprio, dado antes do freeze. Com (b), seguir para A1. A nota do dono de
     2026-11-18 sobre retreinar o DL no Kaggle (seção 2) é uma terceira via condicional a esta falha (ou à
     de P0, abaixo): não
     autoriza nada por si e exigiria ADR nova ou emenda própria, com novo protocolo e novo freeze, e sim
     explícito do dono a cada escrita no Kaggle ou download de saída.
   - Alguma checagem de P0 falha: parar; G1 (P0.1) ou seção 13 (P0.4), e sem A1. A mesma via condicional
     do retreino no Kaggle (nota do dono de 2026-11-18, seção 2) também se liga a esta falha: não autoriza
     nada por si e exige ADR nova ou emenda própria, com sim explícito do dono a cada escrita no Kaggle.
4. **A1.** Texto de aprovação do freeze, usado como `--approval-text` em
   `.venv\Scripts\python.exe scripts\train_dl.py --freeze --approval-text "<texto>"` (saída padrão
   `full_v0`). Proposta, a ajustar pelo dono:
   `Aprovado pelo owner em <AAAA-MM-DD>: recongelar thresholds e shas da validacao do full_v0 apos a perda de data/ em 2026-11-17 (ADR 0007); reproduz o congelamento de 2026-11-16 (ADR 0006 5.1), sem nenhuma mudanca`
5. Rodar o freeze e as checagens F. F.1 a F.4 passam: seguir para A2. Qualquer falha (inclusive a falta de
   referência de F.3): o freeze é apagado, a seção 13 registra, nenhuma avaliação roda, e uma nova
   tentativa de freeze exige diagnóstico e aprovação nova.
6. **A2.** Texto de aprovação da execução, usado como `--approval` em
   `.venv\Scripts\python.exe scripts\evaluate_dl.py --approval "<texto>"` (todos os outros argumentos no
   padrão). Proposta, a ajustar pelo dono:
   `Aprovado pelo owner em <AAAA-MM-DD>: reexecucao deterministica unica da avaliacao de teste do full_v0 apos a perda de data/ em 2026-11-17 (ADR 0007); reproduz a exposicao de 2026-11-16 (ADR 0006 5.2 e 14), sem nenhum ajuste`
7. Rodar `evaluate_dl.py` uma vez, depois R1 e R2.
   - R1 e R2 passam: seguir para A3.
   - R1 ou R2 falha, ou qualquer erro, traceback ou falha a partir da verificação do freeze (tudo o que não
     seja uma recusa do item (i) da seção 3.4, ou na dúvida): conta como abertura; nenhuma segunda
     execução; qualquer passo seguinte exige **ADR nova** (seções 3.4 e 4 R, G2). Recusa do item (i): a
     mesma A2 vale para repetir, com o stderr e o código de saída registrados na seção 13.
8. **A3.** Depois de R1/R2: aceitar ou não o resultado como reprodução, e autorizar a regeneração do
   `docs/results/MANIFEST.md` e o commit (regra de git do repositório).

A1 só é pedido depois de P0 e E passarem (ou depois de Q1 (b)); A2 só depois de F passar. Q2 e Q3 vêm antes
do freeze e antes das checagens E (passo 3), como G3 exige.

## 13. Registro da execução (preencher ao executar)

| passo | data (UTC) | comando | saída literal ou sha | resultado |
|---|---|---|---|---|
| P0.1 a P0.5 | 2026-11-18 ~19:45 | `git diff --stat` (ambos os intervalos) e `git status` nos caminhos da P0.1; hash dos 31 arquivos, labels e manifests (P0.2, P0.3); `prediction_sha256` do relatório (P0.4); `ls` do estado limpo (P0.5). HEAD `5201e40`. | P0.1 vazio e sem alteração; P0.2 31 arquivos, 0 divergentes; P0.3 `labels.parquet` = `labels_sha256` do `delta.json`, `predictions_lgbm_{full,noind}_v0` = `file_sha256` dos manifests, `features.parquet` sem `.manifest.json` (a checagem dele não rodou: pendência registrada); P0.4 20 de 20 entradas não-ensemble casam com `MANIFEST.md`; as 10 de ensemble no relatório são os hashes dos ensembles regenerados (o `--ensemble` reescreveu o relatório; sha atual `5d809b77...` x original `1d1d6efa...`), esperado; P0.5 sem `eval/`, `val_freeze.json` nem `test_opened.json` | PASS (P0.3 com a ressalva de `features.parquet`) |
| `pip freeze` do `.venv` | 2026-11-18 ~19:45 | `.venv\Scripts\python.exe -m pip freeze \| sha256sum` | 78 linhas, sha256 `5522f54ddf95a41a...` | registrado |
| E.1 a E.3 (E.4 não rodada, Q3 = não) | 2026-11-18 ~19:45 | script de leitura em diretório temporário fora do repositório (cópias das 10 predições de rede com sha conferido contra `MANIFEST.md`; sem link); `build_ensemble` por sistema | E.1 10/10 `component_sha256` iguais ao `MANIFEST.md`; E.2 10/10 com `source_id` e as 6 colunas `p_raw_*`/`p_cal_*` iguais (`np.array_equal`, mesmo dtype) e `weight`/`temperature` iguais como texto; E.3 10/10 diferindo só em `created_utc` e `git_sha` | PASS |
| A1 aprovado | 2026-11-18 | texto proposto da seção 12, sem ajuste, aprovado pelo dono na conversa | `Aprovado pelo owner em 2026-11-18: recongelar thresholds e shas da validacao do full_v0 apos a perda de data/ em 2026-11-17 (ADR 0007); reproduz o congelamento de 2026-11-16 (ADR 0006 5.1), sem nenhuma mudanca` | aprovado |
| freeze + F.1 a F.5 | 2026-11-18 ~19:46 | 1ª tentativa: `train_dl.py --freeze --approval-text "<A1>"` sem `--outdir`: recusou antes de gravar (`freeze refused: 40 planned prediction files missing`, caminhos em `data/interim/dataset_100k_v0/`; o padrão do script não é `full_v0`, ao contrário do que a seção 12 diz; nada gravado, teste não lido). 2ª tentativa: o mesmo comando com `--outdir data/interim/full_v0`. Checagens F por script que NÃO chama `verify_freeze`/`load_freeze`. | `val_freeze.json` gravado, 20299 bytes, sha256 `a9d74658d6a6f1f0...` (difere do original `ca360bf6...`, esperado por F3). F.1 labels e delta iguais; F.2 40 sistemas (30 não-ensemble = `MANIFEST.md`, 10 ensemble = arquivos que passaram em E); F.3 LightGBM 40 limiares iguais ao `delta.json` e DL 80 limiares iguais ao `val_purity` do relatório (80 de 80, sem referência faltando); F.4 `constants` = `current_constants()`; F.5 `approval` igual palavra por palavra | PASS |
| A2 aprovado | 2026-11-18 | texto proposto da seção 12, sem ajuste, e comando com `--outdir data/interim/full_v0` (o padrão do script é `dataset_100k_v0`), aprovados pelo dono na conversa | `Aprovado pelo owner em 2026-11-18: reexecucao deterministica unica da avaliacao de teste do full_v0 apos a perda de data/ em 2026-11-17 (ADR 0007); reproduz a exposicao de 2026-11-16 (ADR 0006 5.2 e 14), sem nenhum ajuste` | aprovado |
| `evaluate_dl.py` | 2026-11-18 19:57:32Z (`opened_at`) | `.venv\Scripts\python.exe scripts\evaluate_dl.py --outdir data/interim/full_v0 --approval "<A2>"`, uma única execução | exit 0; stderr vazio; stdout: os 3 caminhos `results_dl.json`, `results_dl.md`, `risk_coverage_dl.csv`; `test_opened.json` criado (`git_sha` `5201e401...`, `freeze_sha256` `a9d74658...`) | executada uma vez |
| R1, R2 | 2026-11-18 | C3 e C4 da ADR 0006 18.4 e 18.5, literais, com `.venv\Scripts\python.exe -I` | R1: as 4 linhas de pares idênticas à "Output on 2026-11-16" (`d`, `ci`, meia-largura, `delta`, `blocks`, `reportable`, veredito de classe, papel e veredito de par); a linha `meta` igual (`292556 597 ... n_boot 1000 seed 20261025`) exceto `git_sha` (`5201e40`) e prefixo do freeze (`a9d74658`), as duas exceções previstas. R2: `cells 108 missing 0` e o conjunto `[(descriptive,False),(front2,False),(front3,False),(front4,False),(headline,True),(sensitivity,True)]` | PASS |
| sha original x sha novo (`val_freeze.json`, `results_dl.*`, `risk_coverage_dl.csv`, `test_opened.json`, 10 ensembles, `train_report_dl.json`) | 2026-11-18 | `sha256sum` dos arquivos gerados contra `MANIFEST.md` (parcial: ensembles e `train_report_dl.json` já registrados acima, bytes diferentes) | `risk_coverage_dl.csv` `42f3dea8...` = original (IDÊNTICO byte a byte); `results_dl.json` `6e3cf051...` x `037e684d...`; `results_dl.md` `68eb98b6...` x `c86a887e...`; `val_freeze.json` `a9d74658...` x `ca360bf6...`; `test_opened.json` `0c8d1cd6...` x `fe77a2e9...` (só mudam por conter `git_sha`, `freeze_sha256` e `opened_at`) | diferenças esperadas; sem divergência de conteúdo em R1/R2 |

Toda invocação de `evaluate_dl.py`, inclusive as que falham, ganha uma linha própria na linha
`evaluate_dl.py` acima, com o stderr **literal** e o código de saída. Em falha, o `test_opened.json`
normalmente não existe e o disco pode parecer "nunca aberto" (seção 3, item 4); por isso o stderr é a
evidência, não o disco. Se a mensagem não for uma das recusas do item (i) da seção 3.4, ou se houver dúvida,
registrar aqui que conta como abertura já ocorrida e que uma ADR nova é exigida; registrar também se existe
em disco algum `test_opened.json` ou arquivo de resultado (nenhum é apagado). Recusa do item (i): registrar
o stderr e o código de saída; a mesma A2 continua valendo para repetir.

## 14. Critérios de aceite (para sair de Proposed)

1. Veredito do `reviewer` sem BLOCKER/MAJOR sobre esta ADR.
2. Aprovação Q0 do dono.

Critérios da execução (não da ADR): P0, E, F, R1 e R2 com resultado "igual" registrado na seção 13, com o
comando e a saída literal (R1 na precisão publicada, seção 4).

## 15. Plano de implementação (por arquivo)

```
FILE: (fora do repositório) diretório temporário do agente, script descartável de checagem   [create]
GOAL: executar P0.2 a P0.5, E.1 a E.3 (e E.4 se Q3) e F.1 a F.3, sem ler rótulos de teste nem calcular métricas de teste, e imprimir uma linha "igual"/"diferente" por condição
EXPORTS: nenhum (script de uso único; regra de três: não vira módulo)
DEPENDS: hashlib, json, pathlib, shutil (só copy2, para E.2), numpy, pyarrow.parquet; astro_triage.models.ensemble.build_ensemble (só para E.2); nenhuma função que abra o teste (proibidos: dl_eval.open_test_set, runner.load_test_set, dl_eval.load_dl_systems, scripts/evaluate_dl.py)
INVARIANTS: somente leitura em data/; nada gravado em data/ nem no repositório; sem junction, symlink ou hardlink; rodar com .venv\Scripts\python.exe -I; F.1 a F.3 só depois de A1 e do freeze
ACCEPTANCE: uma linha por condição P0.2..P0.5, E.1..E.3, F.1..F.3 com "igual" ou "diferente" e o valor comparado; código de saída 0 só se todas forem "igual"
RISK: build_ensemble lê o componente da rede e grava o ensemble no mesmo dl_dir e recusa sobrescrever; o temporário precisa das cópias conferidas dos 10 <net>_full_s<k>, com outdir e lgbm_dir apontando para data/ em leitura
```

```
FILE: docs/results/MANIFEST.md   [modify, gerado por scripts/make_results_manifest.py, depois de A3]
GOAL: registrar os SHA-256 dos arquivos regenerados
EXPORTS: n/a
DEPENDS: scripts/make_results_manifest.py sem mudança
INVARIANTS: a versão de d150490 fica no git como registro da primeira abertura
ACCEPTANCE: o arquivo novo lista val_freeze.json, test_opened.json, results_dl.*, risk_coverage_dl.csv e os 10 ensembles com os shas novos; os 30 shas não-ensemble e o de delta.json são iguais aos da versão d150490
RISK: o script pode não ter campo para citar esta ADR; a ligação fica na seção 13 desta ADR
```

```
FILE: docs/adr/0006-deep-learning-tabular-comparison.md   [modify, tarefa separada, depois de A3]
GOAL: nota datada apontando para a ADR 0007 em 18.1 e 18.4, sem reescrever nada
EXPORTS: n/a
DEPENDS: n/a
INVARIANTS: nenhum texto existente removido; a data da primeira exposição continua 2026-11-16
ACCEPTANCE: existe uma nota datada que cita a ADR 0007, o novo opened_at e o resultado de R1/R2
RISK: confundir "reproduzido" com "final": a nota repete que tudo continua PROVISIONAL
```

Nenhum outro arquivo de código ou de documentação é tocado (em particular, `method_notes.py` e
`html_method.py` não mudam).
