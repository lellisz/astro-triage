// Rotulos estaticos em pt-BR. Nenhum algarismo pode aparecer neste arquivo:
// todo numero e todo texto que depende de dado vem do JSON exportado.

export const siteName = "astro-triage";
export const siteTagline = "Explorador científico";

export type Route = { href: string; label: string };

export const routes: readonly Route[] = [
  { href: "/", label: "Visão geral" },
  { href: "/ceu/", label: "Mapa do céu" },
  { href: "/galaxias/", label: "Galáxias" },
  { href: "/comparacao-dl/", label: "Comparação com DL" },
  { href: "/metodo/", label: "Método" },
];

export const compareHref = "/comparacao-dl/";

export type PageLabels = { eyebrow: string; title: string; lead: string };

export const pages = {
  overview: {
    eyebrow: "Gaia × SDSS · classificação de fontes",
    title: "Classificação de fontes astronômicas",
    lead: "Um panorama do classificador, da distribuição das fontes e da qualidade da avaliação.",
  },
  sky: {
    eyebrow: "Onde o classificador erra",
    title: "Mapa do céu",
    lead: "A taxa de erro por região do céu, para saber onde confiar menos no resultado.",
  },
  galaxies: {
    eyebrow: "Classe de galáxias",
    title: "Galáxias",
    lead: "Pureza e completude das galáxias, com o efeito do prior de quasar.",
  },
  compare: {
    eyebrow: "Redes neurais contra o modelo de árvores",
    title: "Comparação com aprendizado profundo",
    lead: "O que as redes neurais mudam em relação ao classificador entregue.",
  },
  method: {
    eyebrow: "Como os resultados foram obtidos",
    title: "Método",
    lead: "Dados, regras de decisão, limites e glossário.",
  },
} satisfies Record<string, PageLabels>;

export const controls = {
  settings: "Ajustes",
  rule: "Regra",
  prior: "Prior",
  tech: "Modo técnico",
  priorFixed: "Nesta regra o prior é fixo.",
  currentState: "Estado atual",
  techShort: "Modo técnico ativo",
};

export const acessibilidade = {
  mainNav: "Navegação principal",
  settingsRegion: "Ajustes da análise",
  skipToContent: "Ir para o conteúdo",
  seal: "Aviso de resultados provisórios",
};

export const kitLabels = {
  notAvailable: "não disponível",
  tableRegion: "Tabela rolável",
  creditPrefix: "Crédito",
};

export const footer = {
  release: "Versão dos dados",
  note: "Resultados provisórios.",
};

export const overviewLabels = {
  purity: "Pureza de quasar",
  completeness: "Completude de quasar",
  accuracy: "Acurácia geral",
  unavailable: "Esta combinação de regra e prior não está disponível nesta versão dos dados.",
  barsTitle: "Pureza e completude nas condições avaliadas",
  barsSub: "Comparação da regra e dos priors de referência",
  purityChart: "Pureza de quasar",
  completenessChart: "Completude de quasar",
  chartsRegion: "Gráficos de pureza e completude, rolável na horizontal",
  shortcutsTitle: "Continue explorando",
  shortcutsSub: "Outras páginas, mantendo a regra e o prior escolhidos",
  gaiaTitle: "Distribuição no céu",
  gaiaSub: "Referência astronômica real; não é saída do classificador",
  techTitle: "Detalhes técnicos",
  techSub: "Resumo do conjunto de teste",
  nPairs: "Pares cruzados",
  nTest: "Objetos no teste",
  nErrors: "Erros",
  nPixels: "Regiões do céu",
  accuracyBox: "Acurácia",
  qsoHits: "Quasares encontrados",
  qsoTotal: "Quasares de referência",
  classNames: { STAR: "Estrela", GALAXY: "Galáxia", QSO: "Quasar" },
  dataBoxTitle: "Dados do teste",
  classMixTitle: "Composição por classe",
  validationTitle: "Validação",
  validationPurity: "Pureza na validação",
  validationCompleteness: "Completude na validação",
  errorsTitle: "Direções do erro",
  errorsCaption: "Erros por classe verdadeira e classe prevista",
  colTrue: "Classe verdadeira",
  colPred: "Classe prevista",
  colN: "Objetos",
  colShare: "Fração dos erros",
  triageTitle: "Triagem por confiança",
  triageThreshold: "Limiar de confiança",
  triageCoverage: "Cobertura",
  triageAccuracyKept: "Acurácia nos mantidos",
  triageSetAside: "Objetos separados",
  triageErrorsSetAside: "Erros separados",
  triageErrorsTotal: "Erros no total",
  dscTitle: "Comparação com o Gaia DSC",
  dscPopulation: "População",
  dscRows: "Linhas",
  dscScoreChart: "Medida F macro por modelo",
  dscDiffChart: "Diferenças pareadas da medida F macro",
  dscRegion: "Gráficos da comparação com o Gaia DSC, roláveis na horizontal",
  importanceTitle: "Importância dos atributos",
  importanceChart: "Ganho por atributo",
  temperature: "Temperatura de calibração",
  importanceRegion: "Gráficos de importância dos atributos, roláveis na horizontal",
  calibrationTitle: "Calibração",
  calibrationChart: "Diagrama de confiabilidade",
  calibrationX: "Probabilidade média prevista",
  calibrationY: "Frequência observada",
  calibrationDiagonal: "Calibração perfeita",
  calibrationRegion: "Diagrama de confiabilidade, rolável na horizontal",
  confidenceTitle: "Confiança nas previsões",
  confidenceChart: "Objetos por faixa de confiança",
  confidenceX: "Confiança da previsão",
  confidenceY: "Objetos",
  confidenceRight: "Acertos",
  confidenceWrong: "Erros",
  confidenceRegion: "Histograma de confiança, rolável na horizontal",
};

export const skyLabels = {
  regions: "Regiões com dados",
  objects: "Objetos no teste",
  below: "Regiões abaixo do piso",
  maxRate: "Maior taxa de erro",
  mapTitle: "Mapa Mollweide do céu",
  toolbarAria: "Controles do mapa",
  modeGroupAria: "Grandeza exibida no mapa",
  modeRateBtn: "Taxa de erro",
  modeCountBtn: "Quantidade de objetos",
  hideBelow: "Ocultar regiões abaixo do piso",
  contextTitle: "Imagem de contexto",
  contextSub: "O céu inteiro visto pelo Gaia, apenas para referência visual.",
  topTitle: "Regiões com as maiores taxas de erro",
  bandsTitle: "Faixas de latitude galáctica",
  colPosition: "Posição",
  colObjects: "Objetos",
  colErrors: "Erros",
  colRate: "Taxa de erro",
  colRateCi: "Intervalo de confiança",
  colBand: "Faixa",
  colPixels: "Regiões",
};

export const skyMapLabels = {
  mapAria: "Mapa Mollweide do céu: cada marcador é uma região",
  modeRate: "Taxa de erro",
  modeCount: "Contagem de objetos",
  band: "Faixa",
  objects: "Objetos",
  errors: "Erros",
  rate: "Taxa de erro",
  notAvailable: "não disponível",
  belowFloor: "abaixo do piso",
  tooltipRegion: "Detalhes da região",
  legendTitle: "Escala de cinza",
  legendRing: "Anel vazio: região abaixo do piso, sem taxa",
  legendCenter: "Cada marcador é o centro de uma região.",
  atLeast: "≥",
  ringLegendAria: "Símbolo de região abaixo do piso",
  howLabel: "Como ler:",
  hiddenInCountNote: "Regiões abaixo do piso ocultas também neste modo.",
  mapScrollAria: "Mapa do céu, rolável na horizontal",
};

export type SkyMapLabels = typeof skyMapLabels;

export const galaxiesLabels = {
  purityTitle: "Pureza por prior",
  puritySub: "Entre as fontes classificadas como galáxia, quantas são galáxias",
  completenessTitle: "Completude por prior",
  completenessSub: "Entre as galáxias de referência, quantas o modelo encontra",
  networksTitle: "Redes neurais",
  networksSub: "Diferença de pureza em relação ao modelo de árvores",
  errorsTitle: "Onde os erros se concentram",
  errorsSub: "Taxa de erro por magnitude G e por redshift",
  chartPlaceholder: "Gráfico: será desenhado na próxima fase.",
  techTitle: "Notas técnicas",
};

export const compareLabels = {
  unavailable: "Comparação não disponível.",
  unavailableHint: "Não há resultado de aprendizado profundo para esta versão dos dados.",
  result: "Resultado",
  pairs: "Veredito por par",
  verdictGrid: "Grade de veredito",
  verdictGridSoon: "Em construção: a grade com o veredito por classe e por rede aparecerá aqui.",
  absolutePurity: "Pureza absoluta",
  absolutePuritySoon:
    "Em construção: a pureza de cada sistema aparecerá aqui. O intervalo de mínimo a máximo entre sementes não é IC.",
  notCi: "não é IC",
  rules: "Regra de decisão",
  descriptive: "Diferenças descritivas",
  limitations: "Limitações",
};

export const methodLabels = {
  how: "Como o experimento foi feito",
  glossary: "Glossário",
  links: "Referências",
  sources: "Fontes dos dados",
  reading: "Como ler os resultados",
  usage: "Uso do painel",
  rules: "Regras de decisão",
  limits: "Os três limites",
};

export const chartLabels = {
  partialTag: "parcial",
  selectedGlyph: "▸",
};
