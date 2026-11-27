// Figura desenhada pelo matplotlib (astro_triage.web_export.figures) e embutida inline no build
// estatico. O SVG ja chega validado por lib/figures.ts (lido na pagina, no servidor), entao este
// componente nao toca no disco e pode ser usado nas views de cliente.
import type { Figures } from "@/lib/figures";

export function MplFigure({ id, figures, className }: { id: string; figures: Figures; className?: string }) {
  const fig = figures[id];
  if (!fig) throw new Error(`figura ${id} nao foi carregada pela pagina`);
  return (
    <figure className={className} aria-label={fig.aria_label} data-figure={id}>
      <div dangerouslySetInnerHTML={{ __html: fig.svg }} />
      <figcaption>{fig.caption}</figcaption>
    </figure>
  );
}
