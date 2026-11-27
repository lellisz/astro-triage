import { CompareView } from "@/components/pages/CompareView";
import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";
import { compareFigureIds } from "@/lib/figure-ids";
import { loadFigures } from "@/lib/figures";

export default function CompareDlPage() {
  const meta = loadPage("meta");
  const compare = loadPage("compare");
  const { eyebrow, title, lead } = pages.compare;
  return (
    <>
      <PageHeader eyebrow={eyebrow} title={title} lead={lead} showStateChip={false} meta={meta} />
      <CompareView meta={meta} compare={compare} figures={loadFigures(compareFigureIds(compare))} />
    </>
  );
}
