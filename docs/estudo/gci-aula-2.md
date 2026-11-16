# GCI World, Aula 2: Manipulação eficiente de dados com NumPy

Fonte: `lec2_slides.pdf` (80 slides, Matsuo-Iwasawa Lab). Todos os slides foram lidos; nenhum é ilegível. Slides só de título/logo: 1, 4, 11, 19, 46, 80. Os slides 23-24, 63 e 53-56 são variações animadas do mesmo desenho (repetem conteúdo).

Convenção: tudo que não está nos slides vem marcado como **[inferência minha]**. Observação sobre erros nos slides: no slide 6 a linha "maior ou igual" mostra `==` (o correto é `>=`), e "quetient" está escrito errado. São erros de digitação do material.

## 1. Tema em 2 frases

A aula ensina a manipular dados numéricos com NumPy: criar arrays (`np.array`), operar elemento a elemento sem laços (funções universais e broadcasting), agregar (soma, média, máximo) e selecionar partes por índice, fatia, índice avançado e máscara booleana, em 1D e 2D. Isso corresponde à etapa "preparação de dados" do ciclo CRISP-DM mostrada nos slides 12-13. Há também uma revisão de Python (variáveis, operadores, listas, `for`) nos slides 5-10.

## 2. Conceitos-chave

1. **`ndarray` versus lista (slides 14, 21, 22).** Lista aceita tipos misturados e operar entre listas é incômodo; `ndarray` só guarda um tipo, faz aritmética fácil e lida com várias dimensões pelo conceito de *axis*. O NumPy é implementado principalmente em C (rápido, slide 14).
   Analogia: lista é uma gaveta de itens variados; array é uma bandeja de ovos, todas as posições do mesmo tipo e alinhadas.

2. **Funções universais (ufunc) (slides 15, 23-29).** Operam em cada elemento, par a par por posição. `a + b` chama `np.add(a, b)` automaticamente. Valem para `-`, `*` (que **não** é produto escalar nem matricial, o slide avisa), `/`, `exp`, `log`, `sin` (em radianos) e comparações (`==`, `>`, `np.greater`...), que devolvem arrays de verdadeiro/falso. Em listas, `+` concatena e `==` compara a lista inteira (slides 28 e 30).

3. **Broadcasting (slides 31, 49, 50).** Ajuste automático de formas para a operação ser possível. Um escalar é esticado ao tamanho do array. Em N dimensões, dimensões de tamanho 1 são esticadas para casar com a outra; um array 1D de forma (2,) é tratado como (1, 2) antes. Exemplo do slide: (3,1) + (2,) vira (3,2).
   Analogia: um carimbo que o NumPy repete até cobrir a folha.

4. **Funções de agregação (slides 32, 57-61).** Reduzem vários valores a um: `max`, `min`, `sum`, `mean`, `std`. Sem `axis` agregam tudo; com `axis` agregam ao longo de uma direção.

5. **Axis (slides 51-56, 63).** Em 2D, `axis=0` é a direção das linhas (descendo, uma coluna é resumida) e `axis=1` a das colunas (andando pela linha). O slide mostra uma visão "achatada" da lista de listas para ajudar. Efeito prático: `np.max(a, axis=0)` dá o máximo de cada coluna; `axis=1` dá o máximo de cada linha. Truque **[inferência minha]**: o axis indicado é o que *desaparece* da forma do resultado.

6. **Indexação e fatiamento 1D (slides 33-45).** Índices começam em 0; `-1` é o último; lista de índices `a[[1,4,8]]` pega posições soltas. Fatia `[início:fim:passo]`: o fim é **excluído**; início e fim podem ser omitidos (`a[1:]`, `a[:7]`, `a[::2]`). Quiz dos slides: dados de vento a cada 30 min; `a[2]` é 13:00, `a[2:8]` vai de 13:00 até antes de 16:00, `a[::2]` pega de hora em hora.

7. **Indexação 2D, básica e avançada (slides 62-77).** Básica: `a[0, 1]`, `a[0]`, `a[:, 1]`, `a[:2, 1:]` (fatias por eixo). Avançada: listas de índices por eixo, `a[[0,2],[1,3]]` pega os pares (0,1) e (2,3) e devolve forma igual à dos índices. Se os índices tiverem forma (2,1) e (1,2), o broadcasting produz a combinação de todas as linhas com todas as colunas (2x2). Dá para misturar: `a[[0,2], 1:]`.

8. **Indexação booleana (slide 78).** Uma máscara de `True/False` com a mesma forma seleciona os valores `True`. O slide mostra `a % 3 == 0` e destaca que **a saída é 1D**, mesmo vindo de um array 2D.

Também no slide 3: não são cobertos arrays 3D+, modelos complexos de ML em NumPy nem álgebra linear detalhada. Slide 18: funções se consultam na documentação, não se decoram.

## 3. Fórmulas e regras essenciais

Não há fórmulas matemáticas nos slides além das funções de agregação. Abaixo, as regras de sintaxe (texto simples) e uma definição **[inferência minha]** do desvio padrão (o slide só mostra `np.std` e o valor 0,8165 para [1,2,3]).

```
a[inicio:fim:passo]       fim excluido; omitido = vai ate a borda
a[i, j]                   i = axis 0 (linha), j = axis 1 (coluna)
np.max(a, axis=0)         resume cada coluna  -> forma (n_colunas,)
np.max(a, axis=1)         resume cada linha   -> forma (n_linhas,)
a[[i1,i2],[j1,j2]]        elementos (i1,j1) e (i2,j2)
a[mascara]                mascara booleana -> resultado 1D
```

```
std = raiz( soma((x - media)^2) / n )     # np.std usa n por padrao [inferencia minha]
```
Conferindo com o slide: para [1,2,3], media = 2, soma dos quadrados = 2, 2/3 = 0,667, raiz = 0,8165. Bate com o slide 32, o que confirma a divisão por `n` (não `n-1`).

## 4. Perguntas de verificação (sem resposta aqui; gabarito no fim)

1. Se `a = [2, 4]` e `b = [2, 1]` são listas Python, o que dá `a + b`? E se forem arrays NumPy?
2. Dado `v = np.arange(0, 30, 3)` (os dez valores 0, 3, ..., 27), o que retornam `v[1:7:2]`, `v[::2]` e `v[-1]`?
3. Em `m` com forma (3, 4), qual é a forma de `m.max(axis=0)` e qual a de `m.max(axis=1)`? Qual deles dá "a maior nota de cada matéria" se as linhas são matérias (slides 57-60)?
4. Por que `a[[0, 2], [1, 3]]` devolve 2 elementos, enquanto `a[[[0], [2]], [1, 3]]` devolve 4? Explique usando broadcasting.
5. Dado `h = np.array([[2,2,3,6],[0,6,7,9]])`, qual a forma e o conteúdo de `h[h % 3 == 0]`? Por que não preserva a forma 2x4?
6. Um colega escreve `x = np.array([[1],[2],[3]]) + np.array([10, 20])` esperando um erro de forma. O que acontece de fato, e que tipo de bug silencioso isso pode causar ao somar um vetor coluna a um vetor linha por engano? (a segunda parte é **[inferência minha]**)

## 5. Exercício de código (NumPy puro, ~30 linhas)

```python
import numpy as np

# 1) ufunc vs lista (slides 15, 23-28)
a = np.array([2, 4]); b = np.array([2, 1])
print("array a+b:", a + b, "| lista a+b:", [2, 4] + [2, 1])

# 2) broadcasting (slides 31, 49-50)
col = np.array([[1], [2], [3]])      # forma (3,1)
lin = np.array([1, 3])               # forma (2,)
print((col + lin).shape, "\n", col + lin)

# 3) fatiamento 1D: [inicio:fim:passo], fim excluido (slides 38-42)
v = np.arange(0, 30, 3)              # [0,3,...,27]
print(v[1:7:2], v[::2], v[-1], v[[1, 4, 8]])

# 4) axis em 2D (slides 51-61)
m = np.array([[75, 60, 82, 80],
              [90, 73, 75, 85],
              [82, 92, 80, 63]])
print("max geral:", m.max(), "| axis=0:", m.max(axis=0), "| axis=1:", m.max(axis=1))

# 5) indexacao basica e avancada (slides 64-77)
g = np.arange(1, 13).reshape(3, 4)
print(g[0, 1], g[:, 1], g[:2, 1:])
print(g[[0, 2], [1, 3]])             # pares (0,1) e (2,3) -> [2 12]
print(g[[[0], [2]], [1, 3]])         # broadcasting -> 2x2: linhas {0,2} x colunas {1,3}

# 6) indexacao booleana devolve 1D (slide 78)
h = np.array([[2, 2, 3, 6], [0, 6, 7, 9]])
print(h[h % 3 == 0])                 # [3 6 0 6 9]
assert np.array_equal(h[h % 3 == 0], [3, 6, 0, 6, 9])
```

Rodei o código: saídas conferem com os slides (`[4 5]` vs `[2, 4, 2, 1]`; `[ 3  9 15]`; `[90 92 82 85]` para axis=0 e `[82 90 92]` para axis=1; `[ 2 12]`; matriz 2x2 `[[2 4],[10 12]]`; `[3 6 0 6 9]`). Desafio: tire o `[[0],[2]]` e use `[0, 2]` no lugar para ver a diferença; depois repita as agregações com `keepdims=True` e observe a forma **[inferência minha: `keepdims` não está nos slides]**.

## 6. Conexões com o astro-triage

Ligações reais, sem forçar:

- **Pensar em formas e axis é a habilidade básica para as redes neurais do projeto.** Um lote de treino é um array (n_amostras, n_features); a média/desvio por feature para normalizar é `axis=0`. Errar o `axis` (ou deixar o broadcasting casar coluna com linha sem querer, pergunta 6) gera bugs silenciosos, sem erro de forma **[inferência minha]**.
- **Máscaras booleanas (slide 78) para filtrar amostras.** Selecionar objetos por classe ou por condição (por exemplo, rótulo == QSO) é `a[mascara]`. Lembre que a máscara sobre um array 2D devolve 1D; para filtrar *linhas* usa-se uma máscara 1D no eixo 0 **[inferência minha]**.
- **Vetorização no lugar de laços (slides 15, 23-28).** Calcular características derivadas (diferenças de cores, razões) como `a - b` em colunas inteiras é o mesmo mecanismo de ufunc **[inferência minha: o projeto usa pandas/LightGBM, que se apoiam no NumPy, mas não revi o código]**.
- **Cuidado com cortes e fatias.** O "fim excluído" do fatiamento (slides 38-45) é fonte clássica de erro ao dividir dados em treino/validação/teste por posição: `idx[:60]`, `idx[60:80]`, `idx[80:]` não se sobrepõem exatamente por causa disso **[inferência minha]**.

Sem ligação real: revisão de Python (slides 5-10), `np.sin`/`np.exp` trigonométricos, listas 2D de Python.

## Gabarito

1. Listas: concatenação, `[2, 4, 2, 1]`. Arrays: soma elemento a elemento, `[4, 5]`.
2. `v[1:7:2]` = `[3, 9, 15]` (índices 1, 3, 5; o 7 é excluído). `v[::2]` = `[0, 6, 12, 18, 24]`. `v[-1]` = `27`.
3. `axis=0` dá forma `(4,)` (um valor por coluna); `axis=1` dá `(3,)` (um por linha). Com matérias nas linhas, "maior nota de cada matéria" é o máximo de cada linha, ou seja `axis=1`. No slide 60, `np.max(a, axis=1)` dá `[82, 90, 92]`.
4. Índices de mesma forma (2,) e (2,) são pareados posição a posição: (0,1) e (2,3), 2 elementos. Com formas (2,1) e (2,) o broadcasting estica ambos para (2,2): todas as combinações de linhas {0,2} com colunas {1,3}, 4 elementos (slide 74).
5. `[3, 6, 0, 6, 9]`, forma `(5,)`. A máscara marca posições soltas, e o conjunto selecionado não forma uma grade retangular, por isso o NumPy devolve 1D (slide 78).
6. Não dá erro: por broadcasting, (3,1)+(2,) vira (3,2), `[[11,21],[12,22],[13,23]]`. Bug silencioso: se você quisesse somar elemento a elemento dois vetores, recebe uma matriz inesperada que pode seguir adiante no código sem alarme **[inferência minha]**.
