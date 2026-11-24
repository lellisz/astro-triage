import { SkyView } from "@/components/pages/SkyView";
import { PageHeader } from "@/components/shell/PageHeader";
import { pages } from "@/content/labels";
import { loadPage } from "@/lib/data";

export default function SkyPage() {
  const meta = loadPage("meta");
  const sky = loadPage("sky");
  return (
    <>
      <PageHeader
        eyebrow={pages.sky.eyebrow}
        title={sky.texts.title}
        lead={sky.texts.lead}
        showStateChip
        meta={meta}
      />
      <SkyView meta={meta} cards={sky.cards} texts={sky.texts} />
    </>
  );
}
