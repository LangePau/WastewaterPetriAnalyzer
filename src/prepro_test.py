
import cv2
import argparse
from pathlib import Path
from image_loader import load_images_from_folder
from preprocessing import preprocess_batch

def main():
    parser = argparse.ArgumentParser(
        description="Test the preprocessing pipeline on a folder of images."
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        default="data/testdata",
        help="Ordner mit Rohbildern."
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="data/processed",
        help="Ordner zum Speichern der vorverarbeiteten Bilder."
    )
    parser.add_argument(
        "--denoise_method",
        choices=["gaussian", "median"],
        default="gaussian",
        help="Verfahren zur Rauschunterdrückung."
    )
    parser.add_argument(
        "--do_bg_subtract",
        action="store_true",
        help="Ob Background-Subtraction (Median-Blur) angewendet werden soll."
    )
    parser.add_argument(
        "--clahe_clip",
        type=float,
        default=1.0,
        help="CLAHE clipLimit."
    )
    parser.add_argument(
        "--clahe_grid",
        type=int,
        nargs=2,
        metavar=("GX","GY"),
        default=(16,16),
        help="CLAHE tileGridSize (zwei Werte)."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        metavar=("W","H"),
        default=(512,512),
        help="Zielgröße (Breite Höhe) für das Resizing mit Padding."
    )
    args = parser.parse_args()

    # 1) Rohbilder laden
    images, paths = load_images_from_folder(args.input_folder)
    print(f"Loaded {len(images)} images from '{args.input_folder}'.")

    # 2) Preprocessing
    processed = preprocess_batch(
        images,
        denoise_method=args.denoise_method,
        do_bg_subtract=args.do_bg_subtract,
        clahe_clip=args.clahe_clip,
        clahe_grid=tuple(args.clahe_grid),
        target_size=tuple(args.target_size)
    )

    # 3) Ergebnisse abspeichern
    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for img, p in zip(processed, paths):
        out_path = out_folder / p.with_suffix(".png").name
        cv2.imwrite(str(out_path), img)
    print(f"✅ Wrote {len(processed)} images to '{out_folder}'")

if __name__ == "__main__":
    main()
