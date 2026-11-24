import type { Metadata } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import { Shell } from "@/components/shell/Shell";
import { loadPage } from "@/lib/data";
import "./globals.css";

const display = localFont({
  src: [
    { path: "./fonts/instrument-serif-latin-400.woff2", weight: "400", style: "normal" },
    { path: "./fonts/instrument-serif-latin-400-italic.woff2", weight: "400", style: "italic" },
  ],
  variable: "--nf-display",
  display: "swap",
});

const sans = localFont({
  src: [{ path: "./fonts/inter-latin-400-600.woff2", weight: "400 600", style: "normal" }],
  variable: "--nf-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "astro-triage",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  const meta = loadPage("meta");
  return (
    <html lang="pt-BR" className={`${display.variable} ${sans.variable}`}>
      <body>
        <Shell meta={meta}>{children}</Shell>
      </body>
    </html>
  );
}
