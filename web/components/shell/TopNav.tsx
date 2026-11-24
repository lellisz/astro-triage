"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { acessibilidade, routes } from "@/content/labels";
import type { Meta } from "@/lib/schema";
import { readState, withState, type UiState } from "@/lib/ui-state";
import styles from "./TopNav.module.css";

function trimSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

function NavList({ meta, state, pathname }: { meta: Meta; state: UiState; pathname: string }) {
  const current = trimSlash(pathname);
  return (
    <ul className={styles.list}>
      {routes.map((route) => {
        const active = trimSlash(route.href) === current;
        return (
          <li key={route.href}>
            <Link
              href={withState(route.href, state, meta.release)}
              className={active ? `${styles.link} ${styles.active}` : styles.link}
              aria-current={active ? "page" : undefined}
            >
              {route.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function NavWithState({ meta, pathname }: { meta: Meta; pathname: string }) {
  const params = useSearchParams();
  return <NavList meta={meta} state={readState(params, meta)} pathname={pathname} />;
}

export function TopNav({ meta }: { meta: Meta }) {
  const pathname = usePathname();
  return (
    <nav className={styles.nav} aria-label={acessibilidade.mainNav}>
      <Suspense
        fallback={<NavList meta={meta} state={meta.default_state} pathname={pathname} />}
      >
        <NavWithState meta={meta} pathname={pathname} />
      </Suspense>
    </nav>
  );
}
