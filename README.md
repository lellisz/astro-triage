# astro-triage

## English

astro-triage classifies Gaia DR3 sources as STAR, GALAXY or QSO. Labels come from SDSS spectroscopy; the model is LightGBM with calibration, compared against the Gaia DSC classifier as baseline.

> **WARNING: all results are PROVISIONAL.**
> - The `full_v0` evaluation text still carries a stale `PROVISIONAL: 100k sample` header (the numbers are `full_v0`'s; text fix pending).
> - The reporting floor is not met in some cells.
> - The deep-learning comparison protocol (ADR 0006) is not frozen.
> - The population has no magnitude cut (see the ADR 0001 amendment).
>
> Do not cite any number as final.

### Status
- Phase 1 (LightGBM on the `full_v0` dataset, evaluation vs Gaia DSC): done, provisional.
- Layer 2 (Nemotron via NVIDIA API): closed without test evaluation (ADR 0005, Amendment 2).
- Deep-learning comparison: planned, ADR 0006 Proposed.

### How to run
```
.venv\Scripts\python.exe -m pytest
.venv\Scripts\streamlit.exe run src/astro_triage/dashboard/app.py
```

### Data policy
- No raw data in the repository; terms are in `manifests/README.md`.
- Gaia data licence: CC BY-NC 3.0 IGO.
- Citations are pending verification (see `manifests/README.md`).
- The repository is private.

## Português

O astro-triage classifica fontes do Gaia DR3 como STAR, GALAXY ou QSO. Os rótulos vêm da espectroscopia do SDSS; o modelo é LightGBM com calibração, comparado ao classificador Gaia DSC como baseline.

> **AVISO: todos os resultados são PROVISÓRIOS.**
> - O texto da avaliação `full_v0` ainda traz um cabeçalho obsoleto `PROVISIONAL: 100k sample` (os números são do `full_v0`; correção do texto pendente).
> - O piso de reporte não é atingido em algumas células.
> - O protocolo de comparação com deep learning (ADR 0006) não está congelado.
> - A população não tem corte de magnitude (ver a emenda da ADR 0001).
>
> Não cite nenhum número como final.

### Status
- Fase 1 (LightGBM no dataset `full_v0`, avaliação vs Gaia DSC): concluída, provisória.
- Camada 2 (Nemotron via API NVIDIA): encerrada sem avaliação em teste (ADR 0005, Emenda 2).
- Comparação com deep learning: planejada, ADR 0006 Proposed.

### Como executar
```
.venv\Scripts\python.exe -m pytest
.venv\Scripts\streamlit.exe run src/astro_triage/dashboard/app.py
```

### Política de dados
- Nenhum dado bruto no repositório; os termos estão em `manifests/README.md`.
- Licença dos dados Gaia: CC BY-NC 3.0 IGO.
- Citações pendentes de verificação (ver `manifests/README.md`).
- O repositório é privado.
