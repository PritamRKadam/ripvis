"""
coco_utils.py
─────────────
COCO-format annotation generation utilities.

Creates valid COCO instance segmentation JSON files from image/mask pairs.
Compatible with pycocotools for evaluation and with YOLO/MMDetection loaders.
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import cv2
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# COCO dataset builder
# ──────────────────────────────────────────────────────────────────────────────

class COCODatasetBuilder:
    """
    Incrementally build a COCO-format instance segmentation annotation file.

    Usage::

        builder = COCODatasetBuilder(
            dataset_name="RipVIS-Synthetic",
            categories=[{"id": 1, "name": "rip_current", "supercategory": "water"}]
        )
        ann_id = builder.add_image_with_mask(
            image_path="data/synthetic/img_001.png",
            mask_path="data/masks/mask_001.png",
            category_id=1
        )
        builder.save("data/annotations/synthetic_dataset.json")
    """

    def __init__(
        self,
        dataset_name: str = "RipVIS-Synthetic",
        categories: Optional[List[Dict]] = None,
        license_name: str = "CC BY 4.0",
        url: str = "https://ripvis.ai"
    ):
        self._image_id   = 0
        self._ann_id     = 0

        if categories is None:
            categories = [{
                "id": 1,
                "name": "rip_current",
                "supercategory": "water_hazard"
            }]

        self._dataset = {
            "info": {
                "description": dataset_name,
                "url": url,
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "RipVIS Synthetic Pipeline",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [{
                "id": 1,
                "name": license_name,
                "url": "https://creativecommons.org/licenses/by/4.0/"
            }],
            "categories": categories,
            "images": [],
            "annotations": []
        }

    # ── Add images and annotations ─────────────────────────────────────────

    def add_image(
        self,
        image_path: Union[str, Path],
        file_name: Optional[str] = None
    ) -> int:
        """
        Register an image and return its assigned image_id.
        Reads image dimensions from disk.
        """
        image_path = Path(image_path)
        with Image.open(image_path) as img:
            w, h = img.size

        self._image_id += 1
        entry = {
            "id": self._image_id,
            "file_name": file_name or image_path.name,
            "width": w,
            "height": h,
            "license": 1,
            "date_captured": datetime.datetime.now().isoformat()
        }
        self._dataset["images"].append(entry)
        return self._image_id

    def add_annotation_from_mask(
        self,
        image_id: int,
        mask: np.ndarray,
        category_id: int = 1,
        iscrowd: int = 0,
        min_area: int = 500
    ) -> List[int]:
        """
        Add COCO annotation(s) from a binary mask.
        One annotation is created per connected component.

        Args:
            image_id:    ID returned by add_image().
            mask:        (H, W) uint8 binary mask (255 = rip current).
            category_id: Category ID (1 = rip_current by default).
            iscrowd:     0 for instance, 1 for crowd.
            min_area:    Skip components smaller than this pixel count.

        Returns:
            List of annotation IDs added.
        """
        # Label connected components
        n_labels, labels = cv2.connectedComponents(
            (mask > 0).astype(np.uint8)
        )

        added_ids = []
        for label_idx in range(1, n_labels):  # 0 is background
            component_mask = (labels == label_idx).astype(np.uint8) * 255
            area = int((component_mask > 0).sum())

            if area < min_area:
                continue

            # Bounding box [x, y, w, h]
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            x, y, w, h = cv2.boundingRect(contours[0])

            # Segmentation polygons
            polygons = []
            for cnt in contours:
                cnt_squeezed = cnt.squeeze(axis=1)
                if len(cnt_squeezed) < 6:
                    continue
                polygons.append(cnt_squeezed.flatten().tolist())

            if not polygons:
                continue

            self._ann_id += 1
            ann = {
                "id": self._ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": polygons,
                "area": float(area),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": iscrowd
            }
            self._dataset["annotations"].append(ann)
            added_ids.append(self._ann_id)

        return added_ids

    def add_image_with_mask(
        self,
        image_path: Union[str, Path],
        mask: Union[np.ndarray, str, Path],
        category_id: int = 1,
        file_name: Optional[str] = None,
        min_area: int = 500
    ) -> Tuple[int, List[int]]:
        """
        Convenience method: register image + add its mask annotations.

        Args:
            image_path:  Path to the image file.
            mask:        Binary mask as ndarray or path to mask file.
            category_id: COCO category ID.
            file_name:   Override filename in JSON.
            min_area:    Minimum component area to include.

        Returns:
            (image_id, [annotation_ids])
        """
        # Load mask if path given
        if not isinstance(mask, np.ndarray):
            from utils.mask_utils import load_mask
            mask = load_mask(mask)

        image_id = self.add_image(image_path, file_name=file_name)
        ann_ids  = self.add_annotation_from_mask(image_id, mask, category_id,
                                                  min_area=min_area)
        return image_id, ann_ids

    # ── Statistics & I/O ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return summary statistics of the built dataset."""
        return {
            "n_images":      len(self._dataset["images"]),
            "n_annotations": len(self._dataset["annotations"]),
            "categories":    [c["name"] for c in self._dataset["categories"]],
            "total_area":    sum(a["area"] for a in self._dataset["annotations"])
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """Save the COCO JSON file to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._dataset, f, indent=2)
        print(f"Saved COCO annotations: {output_path}")
        stats = self.get_stats()
        print(f"  Images:      {stats['n_images']}")
        print(f"  Annotations: {stats['n_annotations']}")

    def load(self, json_path: Union[str, Path]) -> None:
        """Load an existing COCO JSON into this builder (for appending)."""
        with open(json_path) as f:
            self._dataset = json.load(f)
        # Sync counters
        if self._dataset["images"]:
            self._image_id = max(i["id"] for i in self._dataset["images"])
        if self._dataset["annotations"]:
            self._ann_id = max(a["id"] for a in self._dataset["annotations"])


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_coco_json(json_path: Union[str, Path]) -> bool:
    """
    Basic validation of a COCO JSON file structure.

    Returns True if valid, raises ValueError with details otherwise.
    """
    with open(json_path) as f:
        data = json.load(f)

    required_keys = ["info", "images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required COCO key: '{key}'")

    image_ids   = {img["id"] for img in data["images"]}
    cat_ids     = {cat["id"] for cat in data["categories"]}
    orphaned    = []
    bad_cat     = []

    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            orphaned.append(ann["id"])
        if ann["category_id"] not in cat_ids:
            bad_cat.append(ann["id"])

    if orphaned:
        raise ValueError(f"Annotations with missing image_id: {orphaned[:5]}")
    if bad_cat:
        raise ValueError(f"Annotations with unknown category_id: {bad_cat[:5]}")

    print(f"COCO validation passed: {len(data['images'])} images, "
          f"{len(data['annotations'])} annotations ✓")
    return True


def merge_coco_datasets(
    json_paths: List[Union[str, Path]],
    output_path: Union[str, Path]
) -> None:
    """
    Merge multiple COCO JSON files into one, re-indexing IDs to avoid collisions.

    Args:
        json_paths:  List of COCO JSON file paths to merge.
        output_path: Path to write the merged result.
    """
    merged = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    image_offset = 0
    ann_offset   = 0
    categories_set = False

    for jpath in json_paths:
        with open(jpath) as f:
            data = json.load(f)

        if not categories_set:
            merged["categories"] = data.get("categories", [])
            merged["info"]       = data.get("info", {})
            merged["licenses"]   = data.get("licenses", [])
            categories_set = True

        id_map = {}
        for img in data["images"]:
            new_id = img["id"] + image_offset
            id_map[img["id"]] = new_id
            merged_img = dict(img)
            merged_img["id"] = new_id
            merged["images"].append(merged_img)

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"]       = ann["id"] + ann_offset
            new_ann["image_id"] = id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)

        if data["images"]:
            image_offset = max(i["id"] + image_offset for i in data["images"]) + 1
        if data["annotations"]:
            ann_offset = max(a["id"] + ann_offset for a in data["annotations"]) + 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(json_paths)} COCO files → {output_path}")
    print(f"  Total images:      {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")


if __name__ == "__main__":
    import tempfile, os

    print("Testing coco_utils...")

    builder = COCODatasetBuilder(dataset_name="Test-Dataset")

    # Create a dummy mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 150:400] = 255

    # Create dummy image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_img_path = f.name
    img_arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img_arr).save(tmp_img_path)

    image_id, ann_ids = builder.add_image_with_mask(
        image_path=tmp_img_path,
        mask=mask,
        min_area=100
    )
    print(f"  image_id={image_id}, annotation_ids={ann_ids}")
    stats = builder.get_stats()
    print(f"  Stats: {stats}")

    # Save to temp file and validate
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_json = f.name
    builder.save(tmp_json)
    validate_coco_json(tmp_json)

    os.unlink(tmp_img_path)
    os.unlink(tmp_json)

    print("  coco_utils: OK ✓")
