import { kitLabels } from "@/content/labels";
import { gaiaSkymap } from "@/content/images";
import styles from "./media.module.css";

type GaiaFigureProps = { variant?: "stacked" | "split" };

export function GaiaFigure({ variant = "stacked" }: GaiaFigureProps) {
  const img = gaiaSkymap;
  const split = variant === "split";
  return (
    <figure className={split ? `${styles.figure} ${styles.split}` : styles.figure}>
      {/* eslint-disable-next-line @next/next/no-img-element -- export estatico com images.unoptimized */}
      <img
        className={split ? `${styles.img} ${styles.imgBounded}` : styles.img}
        src={img.src}
        alt={img.alt}
        width={img.width}
        height={img.height}
        loading="lazy"
        decoding="async"
      />
      <figcaption className={split ? `${styles.caption} ${styles.captionSide}` : styles.caption}>
        {img.caption} {kitLabels.creditPrefix}: {img.credit}.
      </figcaption>
    </figure>
  );
}
