import { GalaxiesView } from "@/components/pages/GalaxiesView";
import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import { galaxiesFigureIds } from "@/lib/figure-ids";
import { loadFigures } from "@/lib/figures";

export default function GalaxiesPage() {
  const meta = loadPage("meta");
  const galaxies = loadPage("galaxies");
  const { eyebrow, title, lead } = pages.galaxies;
  return (
    <>
      <PageHeader eyebrow={eyebrow} title={title} lead={lead} showStateChip meta={meta} />
      <GalaxiesView meta={meta} galaxies={galaxies} figures={loadFigures(galaxiesFigureIds())} />
    </>
  );
}
