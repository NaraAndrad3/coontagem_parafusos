import cv2
import matplotlib, numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def annotate_image(img: np.ndarray, blobs: list[dict], count: int) -> np.ndarray:
    """Desenha contornos, centros, rótulos e banner de resultado."""
    out = img.copy()
    
    for i, b in enumerate(blobs):
        n = b.get("estimated", 1)
        # Verde: 1 parafuso | Amarelo: múltiplos (sobrepostos)
        color = (30, 215, 30) if n == 1 else (30, 200, 255)
        
        cv2.drawContours(out, [b["contour"]], -1, color, 2)
        cx, cy = b["center"]
        cv2.circle(out, (cx, cy), 7, color, -1)
        cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)
        
        label = f"x{n}" if n > 1 else str(i + 1)
        cv2.putText(out, label, (cx + 9, cy - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Banner de resultado
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    cv2.putText(out, f"Parafusos: {count}",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 255, 50), 3)
    return out

def draw_heads(img: np.ndarray, heads: list[dict]) -> np.ndarray:
    out = img.copy()

    for i, h in enumerate(heads, start=1):
        x, y, w, h_box = h["bbox"]
        cx, cy = h["center"]

        cv2.rectangle(out, (x, y), (x + w, y + h_box), (255, 0, 255), 2)
        cv2.circle(out, (cx, cy), 5, (255, 0, 255), -1)

        cv2.putText(
            out,
            f"H{i}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

    return out

def save_pipeline_figure(stages: dict, count: int, image_name: str, out_path: str):
    """
    Salva figura com as 5 etapas do pipeline lado a lado.
    Útil para relatório técnico e validação visual.
    """
    names = ["original", "processed", "binary", "mask_clean", "result"]
    titles = [
        "1. Original",
        "2. Gaussian Blur",
        "3. Binarização Otsu",
        "4. Morfologia",
        f"5. Resultado: {count} parafuso(s)",
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.patch.set_facecolor("#0f172a")
    
    for ax, name, title in zip(axes, names, titles):
        img = stages[name]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
    
    fig.suptitle(f"Pipeline de Detecção — {image_name}",
                 color="#7dd3fc", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def save_summary_figure(results: list[dict], out_path: str):
    """
    Figura resumo: todas as imagens com resultado anotado.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor("#0f172a")
    
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        img = cv2.cvtColor(res["result_img"], cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"{res['image']}\n{res['count']} parafuso(s)",
                     color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
    
    fig.suptitle("Sistema de Contagem de Parafusos — Resultados",
                 color="#86efac", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


