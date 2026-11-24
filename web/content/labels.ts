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

export const footer = {
  release: "Versão dos dados",
  note: "Resultados provisórios.",
};
