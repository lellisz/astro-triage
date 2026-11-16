# GCI World, Aula 1: Introdução à Ciência de Dados

Fonte: `lec1_slides.pdf` (34 slides, Matsuo-Iwasawa Lab, Univ. de Tóquio). Todos os slides foram lidos; nenhum é ilegível. Slides só de imagem/logo: 1 (capa), 2/3/11/17 (agendas), 34 (logo final). O slide 23 é uma captura de tela pequena (plataforma "Quri", lista de aulas), legível só em parte.

Convenção: tudo que não está nos slides vem marcado como **[inferência minha]**.

## 1. Tema em 2 frases

A aula argumenta que ciência de dados é "dados + ciência": usar o mesmo ciclo da ciência (hipótese, experimento, análise, revisão) para resolver problemas reais. Ela também apresenta o curso (programação básica, fundamentos de DS, mentalidade de marketing) e um mapa das semanas seguintes.

## 2. Conceitos-chave

1. **Mais dados não garantem mais conhecimento (slide 7).** O slide usa o ciclo CRISP-DM (compreensão do negócio, dos dados, preparação, modelagem, avaliação, expansão) e o exemplo do food truck que vendeu 40 sanduíches. A frase central é "dados importam quando levam a ações".
   Analogia: saber que o caminhão esgotou o estoque é um fato; saber *quando* esgotou e *por quê* é o que permite decidir pedir mais, dar desconto ou mudar o horário.

2. **Pergunte o que está faltando (slide 8).** Existem dados visíveis (banco, arquivo) e dados ocultos (conceitos abstratos ou coisas que não dá para transformar em dado). Decidir só com o que você tem pode levar a decisões erradas (referência citada: "Dark Data", David Hand).
   Analogia: o iceberg, o slide usa exatamente essa imagem.

3. **Alguém precisa ligar o problema aos dados (slide 9).** O cliente diz "queremos melhorar a eficiência de vendas do call center". O cientista de dados não diz "não entendi"; ele pergunta: o que é eficiência? há dados? quais são os KGIs e KPIs? qual a situação atual? objetivo e viabilidade? Depois traduz em algo verificável: "medir recontatos em 30 dias; precisamos de IDs de caso e follow-ups".
   Analogia: um pedido vago ("quero um carro bom") só vira busca útil quando vira critérios (orçamento, uso, prazo).

4. **Três competências do cientista de dados (slide 10).** Conhecimento de domínio/negócio, habilidades de ciência de dados (IA, estatística, processamento de informação) e habilidades de engenharia de dados (usar, implementar e operar de forma útil). O curso cobre as duas últimas.

5. **O ciclo da ciência (slide 12).** Hipotetizar (explicação mais uma previsão), experimentar (evidência por comparação justa), analisar (comparar resultado com previsão), revisar a hipótese. O caso da 7-Eleven (slide 13) aplica isso à previsão de demanda por item, loja e horário, usando vendas recentes, clima e eventos locais.
   Analogia: um cozinheiro que ajusta a receita a cada prato provado.

6. **IA acelera a iteração, mas a pergunta continua sendo sua (slide 14).** Com IA para codar, o ciclo fica mais rápido, mas "traduzir o problema, construir hipóteses e resolver de fato" continua sendo tarefa do humano. Citação do slide (atribuída a Demis Hassabis): fazer a pergunta certa é a parte mais difícil da ciência.

7. **Três chaves (slide 15).** Objetivos guiados por hipótese (resolver problemas de negócio), insights acionáveis (novos projetos e ideias), comparação de métodos (estatística e ciência de dados).

8. **Como estudar (slides 21 e 22).** Não tente decorar tudo: use atalhos, habilidade de pesquisa, ritmo próprio, pergunte e ensine aos colegas. Ao travar, peça à IA a explicação dos conceitos ou mostre a ela a mensagem de erro (e talvez os dados de entrada).

## 3. Fórmulas essenciais

O slide 32 (prévia da semana 12, séries temporais) é a única fórmula da aula:

```
rho_k = soma_{t=k+1..n} (X_t - Xbar) * (X_{t-k} - Xbar)
        ---------------------------------------------
              soma_{t=1..n} (X_t - Xbar)^2
```

- `rho_k`: coeficiente de autocorrelação na defasagem (lag) `k`. Fica entre -1 e 1: perto de 1 é correlação positiva, perto de -1 negativa, perto de 0 nenhuma (slide).
- `X_t`: valor da série no instante `t`. `X_{t-k}`: valor `k` instantes antes.
- `Xbar`: média da série. `n`: número de observações.
- Correlograma: gráfico de `rho_k` contra o lag. Se não decai e continua alto, a série provavelmente tem tendência (slide). Fora da faixa de confiança significa que há autocorrelação naquele lag.

Fórmulas **[inferência minha]** (os slides 26-27 mostram só os números, não as definições):

```
acuracia = acertos / total
recall   = VP / (VP + FN)        # dos positivos reais, quantos achei
precisao = VP / (VP + FP)        # dos que chamei de positivo, quantos eram
```
`VP` verdadeiros positivos, `FN` falsos negativos, `FP` falsos positivos.

Exemplo do slide 27: dois modelos com 90% de acurácia, mas recall de 50% (A, perde 1 de 2 positivos) e 100% (B). "Se perder um positivo custa mais, B é preferível."

## 4. Perguntas de verificação (sem resposta aqui; gabarito no fim)

1. Segundo o slide 7, quando os dados "importam"?
2. Na conversa do call center (slide 9), por que "não entendi o que você quer dizer" é uma resposta ruim do cientista de dados? Cite duas perguntas melhores.
3. Quais são as quatro etapas do ciclo da ciência no slide 12, e qual etapa compara o resultado com a previsão?
4. No slide 26, para que serve separar um conjunto de validação e um de teste, em vez de um só?
5. Dois modelos têm 90% de acurácia, um com recall 50% e outro 100% (slide 27). Quando você escolheria o de recall menor? Que informação você precisa ter para decidir?
6. A autocorrelação de uma série fica em 0,9 em todos os lags de 1 a 12 e não decai. O que isso sugere (slide 32)? Por que esse resultado, sozinho, não prova que a série é "previsível" de forma útil? (a segunda parte é **[inferência minha]**)

## 5. Exercício de código (NumPy puro, ~35 linhas)

Recria o exemplo do slide 27 (acurácia igual, recall diferente) e a divisão treino/validação/teste do slide 26.

```python
import numpy as np

# 20 casos: 18 negativos (0) e 2 positivos (1), como no slide 27
y = np.array([0]*18 + [1]*2)

# Modelo A: 1 falso positivo + 1 verdadeiro positivo; perde 1 positivo
pred_a = np.zeros(20, dtype=int)
pred_a[0] = 1          # falso positivo
pred_a[19] = 1         # acerta 1 positivo (o positivo em 18 e perdido)
# Modelo B: 2 falsos positivos + 2 verdadeiros positivos
pred_b = np.zeros(20, dtype=int)
pred_b[[0, 1]] = 1     # 2 falsos positivos
pred_b[[18, 19]] = 1   # acerta os 2 positivos

def metricas(y, p):
    tp = np.sum((p == 1) & (y == 1))
    fn = np.sum((p == 0) & (y == 1))
    fp = np.sum((p == 1) & (y == 0))
    acc = np.mean(p == y)
    recall = tp / (tp + fn)
    precisao = tp / (tp + fp)
    return acc, recall, precisao

for nome, p in [("A", pred_a), ("B", pred_b)]:
    a, r, pr = metricas(y, p)
    print(f"Modelo {nome}: acuracia={a:.2f} recall={r:.2f} precisao={pr:.2f}")

# Divisao treino/validacao/teste (slide 26): embaralha UMA vez, corta em 60/20/20
rng = np.random.default_rng(0)
idx = rng.permutation(100)
treino, val, teste = idx[:60], idx[60:80], idx[80:]
assert len(set(treino) & set(val) & set(teste)) == 0
print(len(treino), len(val), len(teste))
```

Saída esperada (testei): A: acurácia 0,90, recall 0,50, precisão 0,50; B: acurácia 0,90, recall 1,00, precisão 0,50; depois `60 20 20`. Os números de falsos/verdadeiros positivos foram reconstruídos por mim a partir das contagens do desenho **[inferência minha]**. Desafio: o `assert` só compara a interseção dos três conjuntos; troque por uma checagem de que *nenhum par* de conjuntos se sobrepõe.

## 6. Conexões com o astro-triage

Só incluo as que têm ligação real.

- **Treino / validação / teste (slide 26).** O fluxo do slide (treinar, avaliar no conjunto de validação, voltar a ajustar; o teste só no fim, com dados novos) é o mesmo princípio da disciplina do projeto: validação para decidir (early stopping, calibração, escolha de modelo), teste intocado até o fim. Isso é coerente com o que o CLAUDE.md diz sobre a camada 2 ter sido fechada sem avaliação no teste. Usar o conjunto de teste para ajustar qualquer coisa é vazamento **[inferência minha]**.
- **Acurácia não basta (slide 27).** Em STAR/GALAXY/QSO as classes são desbalanceadas **[inferência minha: a proporção real está nos docs do projeto, não nos slides]**. Um modelo pode ter acurácia alta e recall baixo numa classe rara, como QSO. Olhar recall e precisão por classe é a mesma lição do slide, e o custo de perder um QSO é uma decisão científica, igual ao "se perder um positivo custa mais" do slide.
- **Ciclo hipótese-experimento-análise-revisão (slide 12) e "compare resultados" (slide 33).** O projeto compara LightGBM contra o baseline Gaia DSC e contra redes neurais; essa é a "comparação de métodos" do slide 15. A pré-registração do ADR 0006 é a versão formal de "declare a hipótese e a previsão antes do experimento" **[inferência minha sobre o ADR; não o reli agora]**.
- **Dados ocultos (slide 8).** A população sem corte de magnitude (ADR 0001, segundo o CLAUDE.md) é um exemplo de pergunta "o que está faltando?": os rótulos espectroscópicos do SDSS só existem para quem foi observado **[inferência minha]**.

Sem ligação real: slides sobre indústrias, 7-Eleven, marketing, SQL, clusterização (semana 11): contexto do curso, não do projeto.

## Gabarito

1. Quando levam a ações (decisões concretas: pedir mais, dar desconto, mudar horário).
2. Porque não traduz o pedido vago em algo verificável. Melhores: "o que é eficiência telefônica?", "há dados?", "quais os KGIs/KPIs?", "qual a situação atual?", "quais os objetivos e a viabilidade?". Resultado final do slide: medir recontatos em 30 dias.
3. Hipotetizar, experimentar, analisar, revisar. A etapa que compara resultado com previsão é a análise.
4. O de validação orienta decisões e ajustes durante o desenvolvimento; o de teste só é usado no fim, com dados não vistos, para estimar o desempenho honesto. Se você ajusta com o teste, ele deixa de ser independente **[a segunda frase é inferência minha; o slide mostra o fluxo]**.
5. Escolheria o de recall menor se perder um positivo for barato e falso alarme for caro **[inferência minha]**. Precisa conhecer o custo de cada tipo de erro e a prevalência dos positivos. (O slide só afirma o caso contrário: se perder positivo custa mais, B é melhor.)
6. Sugere que a série tem tendência (slide). Não prova previsibilidade útil porque uma tendência gera autocorrelação alta mesmo sem estrutura preditiva interessante, e a correlação com o passado não diz se o modelo vai acertar o futuro fora da amostra **[inferência minha]**.
