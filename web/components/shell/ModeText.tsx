"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import type { Meta } from "@/lib/schema";
import { readState } from "@/lib/ui-state";

type Props = { meta: Meta; tech: string; simple: string };

function FromUrl({ meta, tech, simple }: Props) {
  return <>{readState(useSearchParams(), meta).tech ? tech : simple}</>;
}

// Texto do cabecalho que muda com o modo tecnico; o cabecalho e server component e nao le a URL.
export function ModeText({ meta, tech, simple }: Props) {
  return (
    <Suspense fallback={<>{meta.default_state.tech ? tech : simple}</>}>
      <FromUrl meta={meta} tech={tech} simple={simple} />
    </Suspense>
  );
}
