import { GraficosView } from "@/components/pages/GraficosView";
import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import { graficosFigureIds } from "@/lib/figure-ids";
import { loadFigures } from "@/lib/figures";

export default function GraficosPage() {
  const meta = loadPage("meta");
  const graficos = loadPage("graficos");
  const { eyebrow, title, lead } = pages.charts;
  return (
    <>
      <PageHeader eyebrow={eyebrow} title={title} lead={lead} showStateChip={false} meta={meta} />
      <GraficosView meta={meta} graficos={graficos} figures={loadFigures(graficosFigureIds(graficos))} />
    </>
  );
}
