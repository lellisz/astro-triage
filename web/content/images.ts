// Imagens estaticas servidas de /public. Dimensoes lidas do cabecalho IHDR do PNG.

export type FigureImage = {
  src: string;
  width: number;
  height: number;
  alt: string;
  caption: string;
  credit: string;
};

export const gaiaSkymap: FigureImage = {
  src: "/gaia/gaia-skymap-monochrome.png",
  width: 1600,
  height: 936,
  alt: "Mapa do céu inteiro observado pelo Gaia, em tons de cinza, com o plano da Via Láctea na horizontal.",
  caption: "Imagem de referência do céu visto pelo Gaia; não representa a saída do classificador.",
  credit: "ESA/Gaia/DPAC",
};
