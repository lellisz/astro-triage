# astro-triage

## English

astro-triage classifies Gaia DR3 sources as STAR, GALAXY or QSO. Labels come from SDSS spectroscopy; the model is LightGBM with calibration, compared against the Gaia DSC classifier as baseline.

> **WARNING: all results are PROVISIONAL.**
> - The reporting floor is not met in some cells.
> - ADR 0006 (deep-learning comparison protocol) is still Proposed: the validation freeze exists, acceptance is open.
> - The population has no magnitude cut (see the ADR 0001 amendment).
>
> Do not cite any number as final.

### Status
- Phase 1 (LightGBM on the `full_v0` dataset, evaluation vs Gaia DSC): done, provisional.
- Layer 2 (Nemotron via NVIDIA API): closed without test evaluation (ADR 0005, Amendment 2).
- Deep-learning comparison (MLP and FT-Transformer vs LightGBM): executed once on the test set under the ADR 0006 protocol; results are aggregates only, provisional, and the test must not be opened again without a new ADR.
- Dashboard (Streamlit): Overview, Sky map, Galaxies, Method and the LightGBM vs deep learning comparison. The Overview, Galaxies and comparison tabs carry a PROVISIONAL seal.

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
> - O piso de reporte não é atingido em algumas células.
> - A ADR 0006 (protocolo da comparação com deep learning) segue Proposed: o freeze da validação existe, a aceitação está aberta.
> - A população não tem corte de magnitude (ver a emenda da ADR 0001).
>
> Não cite nenhum número como final.

### Status
- Fase 1 (LightGBM no dataset `full_v0`, avaliação vs Gaia DSC): concluída, provisória.
- Camada 2 (Nemotron via API NVIDIA): encerrada sem avaliação em teste (ADR 0005, Emenda 2).
- Comparação com deep learning (MLP e FT-Transformer vs LightGBM): executada uma vez no teste pelo protocolo da ADR 0006; resultados só em agregados, provisórios, e o teste não pode ser aberto de novo sem uma nova ADR.
- Dashboard (Streamlit): Visão geral, Mapa do céu, Galáxias, Método e a comparação LightGBM × deep learning. As abas Visão geral, Galáxias e Comparação trazem o selo PROVISÓRIO.

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
