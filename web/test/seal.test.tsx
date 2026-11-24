import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PageHeader } from "@/components/shell/PageHeader";
import type { Meta } from "@/lib/schema";

const meta: Meta = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("./fixtures/fixture_v0/meta.json", import.meta.url)),
    "utf-8",
  ),
);

describe("selo provisorio", () => {
  it("PageHeader renderiza exatamente um role=status com o seal_text", () => {
    const html = renderToStaticMarkup(
      <PageHeader eyebrow="e" title="t" showStateChip={false} meta={meta} />,
    );
    expect(html.match(/role="status"/g)).toHaveLength(1);
    expect(html).toContain(meta.seal_text);
  });
});
