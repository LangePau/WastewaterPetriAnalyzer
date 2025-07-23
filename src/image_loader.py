import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def get_image_paths(
    folder: Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> List[Path]:
    """
    Collect all file paths in `folder` matching the given extensions
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    paths: List[Path] = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            paths.append(file)

    return sorted(paths)


def load_images_from_folder(folder_path: str) -> Tuple[List["np.ndarray"], List[Path]]:
    """
    Load all images from `folder_path` into memory
    Returns a tuple (images, paths)
    """
    folder = Path(folder_path)
    image_paths = get_image_paths(folder)
    images = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path.resolve()}")
        images.append(img)

    return images, image_paths

"""
TESTING

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load all test images from a folder and report count."
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        default="data/testdata",
        help="Path to folder containing test images."
    )
    args = parser.parse_args()

    imgs, paths = load_images_from_folder(args.input_folder)
    print(f"Loaded {len(imgs)} images from '{args.input_folder}'")
    for p in paths:
        print(f" - {p.name}")
"""