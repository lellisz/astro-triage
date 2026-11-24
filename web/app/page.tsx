import { OverviewView } from "@/components/pages/OverviewView";
import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";

export default function OverviewPage() {
  const meta = loadPage("meta");
  const overview = loadPage("overview");
  const { eyebrow, title, lead } = pages.overview;
  return (
    <>
      <PageHeader eyebrow={eyebrow} title={title} lead={lead} showStateChip meta={meta} />
      <OverviewView
        meta={meta}
        rows={overview.rows}
        accuracyNote={overview.accuracy_note}
        technical={overview.technical}
      />
    </>
  );
}
