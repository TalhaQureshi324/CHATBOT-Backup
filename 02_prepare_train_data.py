# import json
# from pathlib import Path
# import importlib

# CONFIG = importlib.import_module("00_config").CONFIG


# def find_json(base_dir, keyword):
#     files = list(Path(base_dir).glob("*.json"))

#     for f in files:
#         name = f.name.lower()
#         if keyword in name:
#             return f

#     raise FileNotFoundError(f"No JSON file found containing: {keyword}")


# def get_image_name(sample):
#     if sample.get("cell_image_cropped"):
#         return Path(sample["cell_image_cropped"]).name

#     if sample.get("cell_image"):
#         return Path(sample["cell_image"]).name

#     raise KeyError("No cell image found in sample.")


# def flatten_english(data, cropped_dir, split_name):
#     rows = []
#     missing_images = []

#     for sample_idx, sample in enumerate(data):
#         image_name = get_image_name(sample)
#         image_path = Path(cropped_dir) / image_name

#         if not image_path.exists():
#             missing_images.append(str(image_path))
#             continue

#         for i in range(1, 21):
#             q = sample.get(f"q{i}")
#             a = sample.get(f"a{i}")

#             if q and a:
#                 q = str(q).strip()
#                 a = str(a).strip()

#                 if q and a:
#                     rows.append({
#                         "source_index": sample_idx,
#                         "image": image_name,
#                         "question": q,
#                         "answer": a,
#                         "qa_index": i,
#                         "split": split_name,
#                         "language": "English"
#                     })

#     return rows, missing_images


# def main():
#     base_dir = Path(CONFIG["base_dir"])

#     train_json = find_json(base_dir, "train")
#     test_json = find_json(base_dir, "test")

#     train_cropped_dir = Path(CONFIG["train_cropped_dir"])
#     test_cropped_dir = Path(CONFIG["test_cropped_dir"])

#     data_dir = Path(CONFIG["data_dir"])
#     data_dir.mkdir(parents=True, exist_ok=True)

#     print(f"Train JSON: {train_json}")
#     print(f"Test JSON:  {test_json}")
#     print(f"Train cropped dir: {train_cropped_dir}")
#     print(f"Test cropped dir:  {test_cropped_dir}")

#     with open(train_json, "r", encoding="utf-8") as f:
#         train_data = json.load(f)

#     with open(test_json, "r", encoding="utf-8") as f:
#         test_data = json.load(f)

#     print("\nFlattening train data...")
#     train_rows, train_missing = flatten_english(
#         train_data,
#         train_cropped_dir,
#         "train"
#     )

#     print("Flattening test data...")
#     test_rows, test_missing = flatten_english(
#         test_data,
#         test_cropped_dir,
#         "test"
#     )

#     print("\nSUMMARY")
#     print("=" * 80)
#     print(f"Train cell samples: {len(train_data)}")
#     print(f"Train VQA rows:     {len(train_rows)}")
#     print(f"Train missing imgs: {len(train_missing)}")
#     print()
#     print(f"Test cell samples:  {len(test_data)}")
#     print(f"Test VQA rows:      {len(test_rows)}")
#     print(f"Test missing imgs:  {len(test_missing)}")

#     if train_missing:
#         print("\nFirst train missing images:")
#         for m in train_missing[:20]:
#             print(m)

#     if test_missing:
#         print("\nFirst test missing images:")
#         for m in test_missing[:20]:
#             print(m)

#     if train_missing or test_missing:
#         raise RuntimeError("Some cropped images are missing. Fix before training.")

#     train_out = data_dir / "train_instances.json"
#     test_out = data_dir / "test_instances.json"

#     with open(train_out, "w", encoding="utf-8") as f:
#         json.dump(train_rows, f, indent=2, ensure_ascii=False)

#     with open(test_out, "w", encoding="utf-8") as f:
#         json.dump(test_rows, f, indent=2, ensure_ascii=False)

#     print("\nSaved:")
#     print(train_out)
#     print(test_out)

#     print("\nSample train row:")
#     print(json.dumps(train_rows[0], indent=2, ensure_ascii=False))

#     print("\nDone.")


# if __name__ == "__main__":
#     main()

import json
from pathlib import Path


def build_image_map(images):
    """
    image_id -> image_file_name
    """
    return {
        img["id"]: img["file_name"]
        for img in images
    }


def flatten_coco_vqa(coco_data, split_name="train"):

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    image_map = build_image_map(images)

    rows = []
    missing_images = set()

    for ann in annotations:

        image_id = ann.get("image_id")

        if image_id not in image_map:
            missing_images.add(image_id)
            continue

        image_path = image_map[image_id]

        # Clean question/answer safely
        question = str(ann.get("Question", "")).replace("Q:", "").strip()
        answer = str(ann.get("Answer", "")).replace("A:", "").strip()

        if len(question) == 0 or len(answer) == 0:
            continue

        rows.append({
            "image_id": image_id,
            "image": image_path,
            "question": question,
            "answer": answer,
            "qa_id": ann.get("id"),
            "category_id": ann.get("category_id"),
            "task": ann.get("task"),
            "split": split_name
        })

    return rows, sorted(list(missing_images))


def main():

    train_json_path = Path("train_coco.json")
    test_json_path = Path("test_coco.json")

    out_dir = Path("data_vqa")
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("Flattening train...")
    train_rows, train_missing = flatten_coco_vqa(train_data, "train")

    print("Flattening test...")
    test_rows, test_missing = flatten_coco_vqa(test_data, "test")

    print("\nSUMMARY")
    print("-" * 60)
    print("Train QAs:", len(train_rows))
    print("Test QAs :", len(test_rows))
    print("Train missing images:", len(train_missing))
    print("Test missing images :", len(test_missing))

    train_out = out_dir / "train_vqa.json"
    test_out = out_dir / "test_vqa.json"

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, indent=2, ensure_ascii=False)

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(train_out)
    print(test_out)


if __name__ == "__main__":
    main()