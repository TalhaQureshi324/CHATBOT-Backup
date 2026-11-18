import json

file1 = "final_cells_morphology_QA_singleQA.json"
file2 = "Q_A_test_large.json"
output_file = "NONVALIDATE_WBC_leukemia_ATT_VQA_TEST.json"

with open(file1, "r") as f:
    coco1 = json.load(f)

with open(file2, "r") as f:
    coco2 = json.load(f)

print(f"File1 images: {len(coco1['images'])}")
print(f"File1 annotations: {len(coco1['annotations'])}")

print(f"File2 images: {len(coco2['images'])}")
print(f"File2 annotations: {len(coco2['annotations'])}")

merged = {
    "images": [],
    "annotations": [],
    "categories": coco1.get("categories", [])
}

new_image_id = 0
new_ann_id = 0


def add_dataset(coco):
    global new_image_id, new_ann_id

    # old image id → new image id
    old_to_new_image = {}

    # Add all images (including duplicates)
    for img in coco["images"]:

        old_id = img["id"]

        merged["images"].append({
            **img,
            "id": new_image_id
        })

        old_to_new_image[old_id] = new_image_id
        new_image_id += 1

    # Add annotations
    for ann in coco["annotations"]:

        new_ann = ann.copy()

        new_ann["id"] = new_ann_id
        new_ann["image_id"] = old_to_new_image[ann["image_id"]]

        merged["annotations"].append(new_ann)

        new_ann_id += 1


add_dataset(coco1)
add_dataset(coco2)

with open(output_file, "w") as f:
    json.dump(merged, f, indent=4)

print("\nFinal merged dataset")
print("Merged images:", len(merged["images"]))
print("Merged annotations:", len(merged["annotations"]))