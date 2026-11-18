import json
from pathlib import Path
from PIL import Image
import importlib

CONFIG = importlib.import_module("00_config").CONFIG


def find_json(base_dir, keyword):
    files = list(Path(base_dir).glob("*.json"))

    for f in files:
        name = f.name.lower()
        if keyword in name:
            return f

    raise FileNotFoundError(f"No JSON file found containing: {keyword}")


def main():
    base_dir = CONFIG["base_dir"]

    train_json = find_json(base_dir, "train")
    train_image_dir = Path(CONFIG["train_image_dir"])
    output_dir = Path(CONFIG["train_cropped_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using train JSON: {train_json}")
    print(f"Train image folder: {train_image_dir}")
    print(f"Output cropped folder: {output_dir}")

    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cropped_count = 0
    skipped_count = 0
    missing_slide_count = 0

    for idx, sample in enumerate(data):
        image_name = sample.get("image_name")
        cell_image = sample.get("cell_image")
        bbox = sample.get("bbox_xyxy")

        if not image_name or not cell_image or not bbox:
            skipped_count += 1
            continue

        slide_path = train_image_dir / image_name
        crop_path = output_dir / cell_image

        if crop_path.exists():
            continue

        if not slide_path.exists():
            missing_slide_count += 1
            if missing_slide_count <= 10:
                print(f"Missing slide image: {slide_path}")
            continue

        try:
            img = Image.open(slide_path).convert("RGB")

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            crop = img.crop((x1, y1, x2, y2))
            crop.save(crop_path)

            cropped_count += 1

        except Exception as e:
            skipped_count += 1
            print(f"Error sample {idx}: {e}")

    print("\nDONE")
    print(f"Total samples: {len(data)}")
    print(f"Cropped new cells: {cropped_count}")
    print(f"Already existing skipped automatically")
    print(f"Missing slide images: {missing_slide_count}")
    print(f"Other skipped: {skipped_count}")
    print(f"Saved in: {output_dir}")


if __name__ == "__main__":
    main()