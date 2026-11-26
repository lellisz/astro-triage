# Roteiro de deploy do site (Vercel, via CLI, build local)

Quem executa: o dono, na máquina dele. Origem: ADR 0008 (item 4 e tarefas W10 a W12).
Estado do ADR 0008 na data deste roteiro: Proposed. Nada abaixo sobe antes de ele ser Accepted.

**Regra de ouro:** todo passo marcado **SÓ COM OK EXPLÍCITO DO DONO** envia dados à Vercel,
autentica numa conta ou altera algo lá. Os passos sem essa marca rodam só na sua máquina.
Nunca cole token em arquivo, comando ou chat: `vercel login` é interativo. O token fica só na
configuração global da CLI, fora do repositório; ao terminar, rode `vercel logout`.

## 0. Termos e limites (Hobby)

- O plano Hobby é só para "personal or non-commercial use" (Termos da Vercel, seção 4).
  O dono declarou uso só de portfólio, sem uso comercial. Se isso mudar, reavaliar antes de subir.
- Limites Hobby (pesquisados em 2026-11-25): 100 GB/mês de Fast Data Transfer, 100 deploys por
  dia, 100 MB por upload via CLI.
- Conferir o tamanho do que será enviado (passo 4) antes do deploy.

## 1. Pré-requisitos

- Node 22 LTS (mínimo do site: 20.9) e npm; Python do projeto (`.venv\Scripts\python.exe`).
- Vercel CLI instalada (`npm i -g vercel@67.0.0`, versão estável em 2026-11-26 via `npm view vercel version`;
  reconferir antes). Instalar a CLI é local; não sobe nada.
- Melhoria futura: no CI (job `web`) as actions usam tags `@v4`, mesma convenção do job `test`;
  fixar por SHA de commit é a recomendação para os dois jobs.
- Conta Vercel do dono. Projeto novo, **sem** integração com Git (um build via Git não teria os
  dados: `web/data/` não está no repositório).

## 2. Checklist pré-deploy (todos "sim" antes de continuar)

- [ ] ADR 0008 com status Accepted.
- [ ] CI verde: job `test` e job `web`.
- [ ] Suíte e lint verdes localmente (comandos abaixo).
- [ ] Auditoria do export: os testes `tests/test_web_export_*.py` passam sobre a release real e
      nenhum JSON contém chave proibida pelo ADR 0008 (somente agregados e regiões HEALPix).
- [ ] OK explícito do dono, na conversa, para o upload (o primeiro deploy já é de produção; ver passo 5).

```powershell
# raiz do repositório
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -q tests/test_web_export_headline.py tests/test_web_export_invariants.py tests/test_web_export_parity.py
cd web
npm ci
npm run lint
npm run typecheck
```

## 3. Exportar a release real e construir (tudo local)

```powershell
# raiz do repositório: grava web/data/full_v0/*.json (pasta ignorada pelo git, fora de data/)
.venv\Scripts\python.exe -m astro_triage.web_export --release full_v0 --out web\data

cd web
$env:ASTRO_RELEASE = "full_v0"
$env:ASTRO_WEB_DATA = (Resolve-Path data).Path
npm run build        # grava web/out
```

Conferir (no diretório `web/`): `out` existe, e os 7 JSONs de página têm `"provisional": true`
(`fmt_samples.json` é só amostra de formatação e não leva o campo). O comando abaixo não deve
listar nada:

```powershell
Get-ChildItem data\full_v0\*.json -Exclude fmt_samples.json |
  Where-Object { -not (Select-String -Path $_.FullName -Pattern '"provisional":\s*true' -Quiet) }
```

Não rode `npm test` depois deste build: `static.test.ts` espera o build da fixture.
Para testes, refaça o build com `ASTRO_RELEASE=fixture_v0` e `ASTRO_WEB_DATA=test/fixtures`.

## 4. Vincular o projeto e gerar a saída da Vercel

```powershell
# vá para web/ a partir de qualquer pasta (o passo 3 já deixou a sessão em web/)
Set-Location (Join-Path (git rev-parse --show-toplevel) "web")

# SÓ COM OK EXPLÍCITO DO DONO: autentica na conta Vercel (abre o navegador; o token fica só na
# configuração global da CLI, fora do repositório)
vercel login

# SÓ COM OK EXPLÍCITO DO DONO: cria/vincula o projeto na Vercel; responder "no" a conectar Git
vercel link

# Local, mas depende do passo anterior (lê .vercel/project.json): gera web/.vercel/output.
# ATENÇÃO: `vercel build` refaz o `next build`. As variáveis do passo 3 só valem na mesma sessão
# do PowerShell; defina-as de novo aqui, imediatamente antes, na mesma sessão.
# O `vercel build` pode pedir para baixar as configurações do projeto (download, não upload).
$env:ASTRO_RELEASE = "full_v0"
$env:ASTRO_WEB_DATA = (Resolve-Path data).Path
vercel build --prod

# Local: o envio será no máximo este tamanho (limite Hobby de 100 MB por upload CLI)
"{0:N1} MB" -f ((Get-ChildItem .vercel\output -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
```

`web/.vercel/` já está no `.gitignore` da raiz. Confirme com `git status` que nada de `.vercel/`
aparece.

Checklist pós-build:

- [ ] `out` (ou `.vercel\output\static`), no diretório `web/`, corresponde à release certa:
      `data\full_v0\meta.json` tem a release `full_v0`, e
      `Select-String -Path out\*.html -Pattern fixture_v0 -List` não acha nada. Se achar `fixture_v0`, as variáveis não valeram: refaça o passo.

**A confirmar no primeiro deploy (não provado pela documentação):**

1. `vercel build` com `output: "export"` (projeto Next.js estático). Sem exemplo oficial
   confirmado. Os "fast static deployments" da Vercel NÃO aceitam JavaScript nem `vercel.json`
   na raiz, por isso este roteiro usa o fluxo padrão com `vercel build`. Se falhar, parar e
   reportar; não improvisar.
2. Os `headers` de `web/vercel.json` valem com `vercel build` + `--prebuilt` num export
   estático, e são convertidos para `routes` no `.vercel/output/config.json`. Conferir
   `.vercel/output/config.json` após o build (procurar `Content-Security-Policy`) e os
   cabeçalhos reais após o deploy (passo 7). Se não valerem, o site funciona sem cabeçalhos;
   decidir com o dono se aceita ou se muda a abordagem.
3. Logo após o passo 5, conferir no painel da Vercel (Domains / Deployments) que NENHUM domínio
   de produção (`<projeto>.vercel.app`) aponta para o deploy. Se apontar, parar e usar
   `vercel remove <URL>` (**SÓ COM OK EXPLÍCITO DO DONO**).

## 5. Primeiro deploy (de produção, sem domínio)

O PRIMEIRO deploy de um projeto novo é sempre de produção, e `--skip-domain` só pode ser usado
com `--prod` (vercel.com/docs/cli/deploy). Este deploy já é de produção: apenas não recebe o
domínio de produção, ficando só a URL do deploy. Essa URL é pública: qualquer pessoa com o link
acessa (a Deployment Protection padrão de projetos novos pode exigir login, mas não conte com
isso). Por isso o build do passo 4 usou `vercel build --prod`.

```powershell
# SÓ COM OK EXPLÍCITO DO DONO: envia web/.vercel/output para a Vercel (produção, sem domínio)
vercel deploy --prebuilt --prod --skip-domain
```

Anote a URL impressa. Faça a verificação do passo 7 nela.

## 6. Promover para produção

```powershell
# SÓ COM OK EXPLÍCITO DO DONO: promove o deploy já verificado para o domínio de produção
vercel promote <URL-do-deploy>
```

Deploys seguintes, quando o dono autorizar de novo, repetem o fluxo completo do primeiro:
exportar a release (passo 3), auditar o export (passo 2), `vercel build --prod` (passo 4),
`vercel deploy --prebuilt --prod --skip-domain` (**SÓ COM OK EXPLÍCITO DO DONO**), verificar
como no passo 7 e só então `vercel promote` (passo 6, **SÓ COM OK EXPLÍCITO DO DONO**). Nenhuma
release nova vai direto ao domínio público.

## 7. Verificação pós-deploy

- [ ] As 6 rotas abrem e mostram o selo PROVISÓRIO: `/`, `/ceu/`, `/galaxias/`,
      `/comparacao-dl/`, `/graficos/`, `/metodo/`.
- [ ] A release exibida é `full_v0` (não `fixture_v0`).
- [ ] Console do navegador (F12, aba Rede e Console): nenhuma requisição para outro domínio
      e nenhum erro de Content-Security-Policy.
- [ ] Cabeçalhos presentes (conferência no navegador; não cria segredo nem altera o projeto).
      Projetos novos vêm com Deployment Protection ligada: a URL do deploy provavelmente exige
      login da Vercel e um `curl -sI` anônimo não mostraria os cabeçalhos do site. Confira num
      navegador logado na Vercel (DevTools, aba Rede, abrindo a URL do deploy). NÃO use
      `vercel curl` nem crie `protection-bypass`/Protection Bypass for Automation: isso grava
      um segredo permanente no projeto. Nunca desligue a proteção para testar. Mesmo assim,
      trate a URL como pública por padrão.

```powershell
# alternativa anônima, só útil se a proteção estiver desligada
curl.exe -sI <URL-do-deploy>/ | Select-String -Pattern "content-security-policy|x-frame-options|x-content-type-options|referrer-policy|permissions-policy"
```

Se faltar cabeçalho, ver o item 2 do passo 4. Se uma rota quebrar por causa da CSP, o relatório
do console diz qual diretiva; reportar em vez de afrouxar às cegas.

## 8. Rollback

O site é estático e sem estado: reverter é trocar qual deploy responde, sem migração.

```powershell
# SÓ COM OK EXPLÍCITO DO DONO: volta a produção para o deploy anterior (instantâneo)
vercel rollback

# SÓ COM OK EXPLÍCITO DO DONO: remove um deploy específico (deploy defeituoso ou indesejado)
vercel remove <URL-do-deploy>
```

Se o primeiro deploy saiu com a release errada: `vercel remove <URL>` (**SÓ COM OK EXPLÍCITO
DO DONO**) e refazer a partir do passo 3. Se o projeto inteiro deve sumir, apague-o no painel da Vercel (ação do dono; fora do
IaC porque o projeto não é gerenciado por código).

## 9. Custo

Ao terminar todo o trabalho com a CLI, rode `vercel logout`.

US$ 0 no plano Hobby (sem recurso pago). Monitorar o uso no painel da Vercel contra os limites
do passo 0. Qualquer upgrade para plano pago exige novo ADR (CLAUDE.md: sem provedor pago).

## Content-Security-Policy: trade-off de `'unsafe-inline'`

O export estático do Next.js 16 emite scripts inline (bootstrap do React/Turbopack e dados
`self.__next_f.push(...)`), 28 nas páginas do build de fixture, e o conteúdo muda a cada página
e a cada release. Há também 3 `<style>` (páginas de erro) e 17 atributos `style=""`.

- Com CSP por hash: seria preciso calcular o hash de cada script inline de cada página a cada
  build e gerar o `vercel.json` depois dele. Um export estático sem servidor não tem nonce por
  requisição. Custo alto e frágil para um site de agregados sem entrada de usuário.
- Escolha: `script-src 'self' 'unsafe-inline'` e `style-src 'self' 'unsafe-inline'`. Isso
  enfraquece a proteção contra XSS injetado no HTML. O risco residual é baixo porque o site não
  tem formulário, cookie, login nem renderiza entrada de usuário, e o HTML é gerado só de JSON
  próprio. O restante da política continua restritivo: `default-src 'self'`,
  `connect-src 'self'` (sem chamada externa), `img-src 'self'`, `font-src 'self'`,
  `object-src 'none'`, `frame-ancestors 'none'`.
- Os links externos da página Método (DOI, GitHub, ESA, SDSS) são navegação, não são afetados
  pela CSP.
- Revisão futura: se o site ganhar entrada de usuário ou servidor, migrar para nonce/hash.
