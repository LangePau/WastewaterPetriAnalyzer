# src/image_loader.py

import cv2
from pathlib import Path
from typing import List, Tuple

def get_image_paths(
    folder: Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> List[Path]:
    """
    Collect all file paths in `folder` matching the given extensions (case-insensitive)
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    paths: List[Path] = []
    # Only iterate over the top-level items in the folder
    for file in folder.iterdir():
        # Check files only, and compare suffix in lowercase
        if file.is_file() and file.suffix.lower() in extensions:
            paths.append(file)

    # Sort for reproducibility
    return sorted(paths)


def load_images_from_folder(folder_path: str) -> Tuple[List["numpy.ndarray"], List[Path]]:
    """
    Load all images from `folder_path` into memory (non-recursive).
    Returns a tuple (images, paths).
    """
    folder = Path(folder_path)

    # print(f"[DEBUG] Using input folder: {folder.resolve()}")

    # 1) Collect image paths
    image_paths = get_image_paths(folder)
    
    # print(f"[DEBUG] Found {len(image_paths)} image(s): {[p.name for p in image_paths]}")

    images = []
    for path in image_paths:
        # 2) Load each image, preserving bit-depth/channels
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path.resolve()}")
        images.append(img)

    # 3) Return both the arrays and their file paths
    return images, image_paths

"""
TESTING
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load all test images from a folder (non-recursive) and report count."
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