import os
import xml.etree.ElementTree as ET
import json
import csv
from collections import defaultdict
import shutil

def parse_xml(xml_path):
    """
    Parse your existing XML annotation; return:
      - width, height
      - list of objects: each is dict {label, xmin, ymin, xmax, ymax}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    # Depending on your XML structure: here we assume <outputs><object><item> etc
    outputs = root.find("outputs")
    if outputs is not None:
        obj_block = outputs.find("object")
        if obj_block is not None:
            for item in obj_block.findall("item"):
                label = item.find("name").text
                bbox = item.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                objects.append({
                    "label": label,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
    return width, height, objects

def export_unified_csv(split_csv_path, anno_dir, image_dir, class_map, output_csv_path):
    """
    Using your split CSV (which has image file names + bounding box entries),
    and XMLs + images, produce a unified CSV with columns:
      image_path, xmin, ymin, xmax, ymax, label_id, label_str, width, height, patient_id
    """
    with open(split_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    out_rows = []
    for row in rows:
        # get image filename
        # this depends on CSV; maybe column 'image' or 'filename'
        img_name = row.get("image", row.get("filename", None))
        if img_name is None:
            continue
        # relative image path
        image_path = os.path.join(image_dir, img_name)
        # XML path
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(anno_dir, xml_name)
        if not os.path.exists(xml_path):
            print(f"XML not found for {img_name}, skipping")
            continue

        width, height, objs = parse_xml(xml_path)
        # for each object
        for obj in objs:
            label_str = obj["label"]
            label_id = class_map.get(label_str, None)
            if label_id is None:
                # assign new id or skip
                continue
            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]
            # patient id from filename, e.g. split by underscore or directory
            patient_id = os.path.splitext(img_name)[0].split("_")[0]

            out_rows.append({
                "image_path": image_path,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label_id": label_id,
                "label": label_str,
                "width": width,
                "height": height,
                "patient_id": patient_id
            })

    # write CSV
    with open(output_csv_path, "w", newline='') as f:
        fieldnames = ["image_path","xmin","ymin","xmax","ymax","label_id","label","width","height","patient_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote unified CSV: {output_csv_path}, entries: {len(out_rows)}")

def export_coco_json(split_name, split_csv_path, anno_dir, image_dir, class_map, output_json_path, start_image_id=1, start_ann_id=1):
    """
    Create COCO JSON annotations for one split.
    """
    images = []
    annotations = []
    categories = []
    # build categories list
    for label_str, label_id in class_map.items():
        categories.append({
            "id": label_id,
            "name": label_str,
            "supercategory": "none"
        })

    ann_id = start_ann_id
    img_id = start_image_id
    # To avoid duplicates of images, map image name → image_id
    image_id_map = {}

    with open(split_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        img_name = row.get("image", row.get("filename"))
        if img_name is None:
            continue
        if img_name not in image_id_map:
            # new image entry
            xml_name = os.path.splitext(img_name)[0] + ".xml"
            xml_path = os.path.join(anno_dir, xml_name)
            if not os.path.exists(xml_path):
                print(f"XML missing for {img_name}, skipping image from COCO")
                continue
            width, height, objs = parse_xml(xml_path)
            image_rec = {
                "file_name": os.path.join(image_dir, img_name),
                "height": height,
                "width": width,
                "id": img_id
            }
            images.append(image_rec)
            image_id_map[img_name] = img_id
            img_id += 1

        # find the objects for this image (re-parse xml or use objs from parse)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(anno_dir, xml_name)
        if not os.path.exists(xml_path):
            continue
        width, height, objs = parse_xml(xml_path)
        for obj in objs:
            label_str = obj["label"]
            label_id = class_map.get(label_str, None)
            if label_id is None:
                continue
            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]
            # COCO bbox format: [x, y, width, height]
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            coco_ann = {
                "id": ann_id,
                "image_id": image_id_map[img_name],
                "category_id": label_id,
                "bbox": [xmin, ymin, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            }
            annotations.append(coco_ann)
            ann_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json_path, "w") as f:
        json.dump(coco_json, f)
    print(f"Wrote COCO json: {output_json_path}, {len(images)} images, {len(annotations)} annotations")

def export_yolo_txts(split_csv_path, anno_dir, image_dir, class_map, output_labels_dir, rel_image_dir):
    """
    For YOLO format: one .txt per image; each line: <class_id> <x_center> <y_center> <width_norm> <height_norm>
    rel_image_dir = base folder for images in the dataset for use in filenames in TXT or matching
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    with open(split_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # grouping by image
    rows_by_image = defaultdict(list)
    for row in rows:
        img_name = row.get("image", row.get("filename"))
        if img_name is None:
            continue
        rows_by_image[img_name].append(row)

    for img_name, img_rows in rows_by_image.items():
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(anno_dir, xml_name)
        if not os.path.exists(xml_path):
            print(f"Missing xml for YOLO txt: {img_name}")
            continue

        width, height, objs = parse_xml(xml_path)
        if width is None or height is None:
            print(f"Missing size for {img_name}")
            continue

        out_txt_path = os.path.join(output_labels_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(out_txt_path, "w") as fw:
            for row in img_rows:
                label_str = row.get("class", row.get("label", row.get("name")))
                label_id = class_map.get(label_str, None)
                if label_id is None:
                    continue
                xmin = int(row.get("xmin", row.get("x_min")))
                ymin = int(row.get("ymin", row.get("y_min")))
                xmax = int(row.get("xmax", row.get("x_max")))
                ymax = int(row.get("ymax", row.get("y_max")))

                # YOLO normalized center + width height
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                w_norm = (xmax - xmin) / width
                h_norm = (ymax - ymin) / height

                fw.write(f"{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    print(f"YOLO txts written to {output_labels_dir}")


if __name__ == "__main__":
    # Example usage (you’d set these paths accordingly)
    class_map = {"异常": 1}  # add others if needed

    # Paths
    anno_dir = "anno_copy"
    image_dir = "JPEGImages"
    splits = {
        "train": "csvfiles/fold1/train.csv",
        "val": "csvfiles/fold1/val.csv",
        "test": "csvfiles/test.csv"
    }

    os.makedirs("exports", exist_ok=True)
    for split_name, split_csv in splits.items():
        # unified csv
        export_unified_csv(split_csv, anno_dir, image_dir, class_map, os.path.join("exports", f"{split_name}_unified.csv"))

        # COCO json
        export_coco_json(split_name, split_csv, anno_dir, image_dir, class_map, os.path.join("exports", f"{split_name}_coco.json"))

        # YOLO txts
        labels_out_dir = os.path.join("exports", f"{split_name}_yolo_labels")
        export_yolo_txts(split_csv, anno_dir, image_dir, class_map, labels_out_dir, image_dir)
