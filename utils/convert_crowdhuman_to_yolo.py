import json
import os
from tqdm import tqdm
import cv2

# Input annotation files
train_odgt = "annotation_train.odgt"
val_odgt = "annotation_val.odgt"

# YOLO dataset paths
train_img_dir = "yolo_dataset/images/train"
val_img_dir = "yolo_dataset/images/val"

train_lbl_dir = "yolo_dataset/labels/train"
val_lbl_dir = "yolo_dataset/labels/val"

os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

def process_odgt(odgt_path, img_dir, lbl_dir):
    with open(odgt_path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line)
        img_id = data["ID"]

        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        label_path = os.path.join(lbl_dir, img_id + ".txt")
        out = open(label_path, "w")

        for gt in data["gtboxes"]:
            if gt["tag"] != "person":
                continue

            # CrowdHuman uses fbox for full person bounding box
            if "fbox" not in gt:
                continue

            x, y, w, h = gt["fbox"]

            # convert to YOLO center format
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            nw = w / W
            nh = h / H

            # Write: class xc yc w h
            out.write(f"0 {xc} {yc} {nw} {nh}\n")

        out.close()

print("Processing TRAIN...")
process_odgt(train_odgt, train_img_dir, train_lbl_dir)

print("Processing VAL...")
process_odgt(val_odgt, val_img_dir, val_lbl_dir)

print("DONE!")
