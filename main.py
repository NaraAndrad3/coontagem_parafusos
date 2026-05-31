import cv2
import json
from pathlib import Path

from src.detector import process_image
from src.visualization import save_pipeline_figure, save_summary_figure


def main():
    img_dir = Path("data/imgs")
    out_dir = Path("results/imgs_anotadas")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("img*.jpg"))
    results = []

    print("=" * 60)
    print("  SISTEMA DE CONTAGEM DE PARAFUSOS")
    print("  Visão Computacional Clássica — OpenCV")
    print("=" * 60)

    for img_path in images:
        res = process_image(str(img_path))
        results.append(res)

        cv2.imwrite(
            str(out_dir / f"result_{img_path.stem}.jpg"),
            res["result_img"]
        )

        save_pipeline_figure(
            res["stages"],
            res["count"],
            res["image"],
            str(out_dir / f"pipeline_{img_path.stem}.png")
        )

        print(f"\n  {img_path.name}")
        print(f"    Blobs válidos : {res['blobs']}")
        print(f"    Corpos        : {res['body_count']}")
        print(f"    Cabeças       : {res['head_count']}")
        print(f"    Contagem final: {res['count']} parafuso(s)")

    save_summary_figure(
        results,
        str(out_dir / "summary_results.png")
    )

    print("\n" + "=" * 60)
    print("  SUMÁRIO FINAL")
    print("=" * 60)

    for r in results:
        bar = "█" * min(r["count"], 30)
        print(f"  {r['image']:12s}  {r['count']:3d}  {bar}")

    summary = [
        {
            "image": r["image"],
            "count": r["count"],
            "body_count": r["body_count"],
            "head_count": r["head_count"],
            "blobs": r["blobs"],
        }
        for r in results
    ]

    with open(out_dir / "counts.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Arquivos salvos em {out_dir}/")
    print("  ✅ Concluído.")


if __name__ == "__main__":
    main()