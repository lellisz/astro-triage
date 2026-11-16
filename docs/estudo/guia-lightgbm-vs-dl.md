# Guia de estudo: LightGBM contra deep learning no astro-triage

Objetivo: conseguir contar o projeto em uma entrevista, de forma simples e correta.
Todos os números do projeto são PROVISÓRIOS. Não os apresente como finais.

## Camada 1 — O projeto em 30 segundos (fala de entrevista)

"Eu classifico fontes do Gaia DR3 em estrela, galáxia ou quasar, usando os rótulos espectroscópicos
do SDSS como verdade. Primeiro construí um modelo forte com LightGBM e calibrei suas probabilidades.
Depois, em vez de assumir que deep learning seria melhor, escrevi antes, em um documento
(ADR 0006), a regra que decide quem ganha. Só abro o conjunto de teste uma vez, depois de
congelar tudo na validação."

Três ideias para guardar: **baseline forte**, **regra escrita antes**, **teste aberto uma vez**.

## Camada 2 — Por que comparar LightGBM com deep learning

- Em dados **tabulares** (colunas numéricas), modelos de árvores como o LightGBM costumam ser muito
  competitivos. Redes neurais nem sempre ganham. Por isso o projeto trata o LightGBM como o
  adversário a vencer, não como um detalhe.
- Comparar é fácil de fazer errado. Dá para "descobrir" que a rede ganha por acaso, ou porque o
  teste vazou para o ajuste. O ADR 0006 existe para impedir isso.
- A regra de decisão do ADR 0006:
  1. Métrica principal: **pureza** de QSO e de GALÁXIA quando a **completude** é fixada em 0,80
     (0,90 é secundária).
  2. A rede só ganha se superar o LightGBM por mais que **δ**, que é 2 desvios-padrão de 5 sementes
     do próprio LightGBM, medidos na validação. Ou seja: a diferença tem que ser maior que o
     "ruído" natural do LightGBM.
  3. Além disso, o intervalo de confiança por **bootstrap por blocos do céu** tem que ficar
     acima de zero nas duas classes.
  4. **Empate: fica o LightGBM.**
  5. O teste é aberto **uma única vez**.

Pergunta de entrevista: "Por que a regra de empate favorece o LightGBM?" Resposta: porque ele já é
mais barato e mais simples. A rede precisa provar que vale o custo extra.

## Camada 3 — Como a rede treina

**As duas redes** (configurações fixas em `src/astro_triage/models/dl_nets.py`):
- **MLP**: 3 camadas escondidas de 512 neurônios, ativação ReLU, dropout 0,1.
- **FT-Transformer**: cada coluna vira um "token" (um vetor); 3 blocos de atenção com 8 cabeças;
  um token especial [CLS] resume a linha e alimenta a cabeça que dá os 3 logits (estrela/galáxia/quasar).

**O loop** (`src/astro_triage/models/dl_train.py`):
1. A cada época, os dados de treino são embaralhados e processados em lotes de 1024.
2. Perda: entropia cruzada, sem pesos de classe.
3. Depois de cada época, mede-se a perda na **validação**. Se melhorou (estritamente), guarda os
   pesos como "melhores". Se não, conta uma época sem melhora.
4. **Early stopping**: para quando as épocas sem melhora chegam à "paciência". Os pesos finais são os
   melhores, não os últimos.
5. **Checkpoint**: a cada época grava `last.pt` (modelo, otimizador, contadores e estado dos
   geradores aleatórios). Se a sessão cair, retoma do ponto e chega **ao mesmo resultado, bit a bit**.

**Depois do treino**:
- **Temperatura**: uma constante T ajustada na validação para que as probabilidades sejam
  honestas (calibração).
- **Ensemble**: mistura LightGBM e rede com um peso w escolhido numa grade, **só na validação**.

**A regra de ouro (vazamento):** o pré-processamento, o early stopping, a temperatura e o peso do
ensemble usam só treino e validação. O teste só passa pelo modelo já congelado.

## Camada 4 — Glossário rápido

| Termo | Em uma frase |
|---|---|
| Pureza (precisão) | Entre os que o modelo chamou de quasar, quantos são de fato. |
| Completude (recall) | Dos quasares reais, quantos o modelo achou. |
| Prior | A proporção esperada de cada classe no céu (quasares são raros). |
| Calibração | Fazer "80%" significar que acerta cerca de 80% das vezes. |
| Temperatura | Constante que suaviza ou afia as probabilidades. |
| Early stopping | Parar o treino quando a validação deixa de melhorar. |
| Vazamento | Informação do teste influenciando o ajuste do modelo. |
| Bootstrap por blocos | Reamostrar regiões do céu, não objetos soltos, porque vizinhos são parecidos. |
| Semente | Número que fixa a aleatoriedade; 5 sementes medem o "ruído" do modelo. |

## Camada 5 — Perguntas de entrevista para treinar

1. Por que não usar só a acurácia para comparar os modelos?
2. O que é vazamento e como o projeto o evita?
3. Por que 5 sementes do LightGBM definem o δ?
4. Por que a validação serve para ajustar e o teste não?
5. O que o checkpoint garante que o "salvar o modelo" simples não garante?
6. Por que o resultado de um quasar raro pode ter pureza baixa mesmo com acurácia alta?

## Camada 6 — Onde olhar no código

| Para entender | Leia |
|---|---|
| A pergunta científica | `docs/adr/0006-deep-learning-tabular-comparison.md` |
| As redes | `src/astro_triage/models/dl_nets.py` e `tests/test_dl_nets.py` |
| O loop e a retomada | `src/astro_triage/models/dl_train.py` e `tests/test_dl_train_loop.py` |
| O ensemble | `src/astro_triage/models/ensemble.py` |
| O congelamento | `src/astro_triage/evaluation/dl_val.py` |
