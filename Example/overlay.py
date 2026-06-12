import os
import cv2
import numpy as np
import random
import argparse
from skimage.exposure import match_histograms

# ===== Default hyperparameters =====
MAX_BG_TRIES_PER_REF = 8

# Scale candidates (from large to small)
SCALES = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2, 0.16, 0.125]

# Number of position sampling trials
TRIALS_FIRST_SCALE = 3000
TRIALS_OTHER_SCALES = 600


# ===== Utility functions =====
def match_texture_statistics(obj_patch, bg_patch):
    obj_patch = obj_patch.astype(np.float32)
    bg_patch = bg_patch.astype(np.float32)
    for c in range(3):
        obj_mean, obj_std = np.mean(obj_patch[:, :, c]), np.std(obj_patch[:, :, c]) + 1e-5
        bg_mean, bg_std = np.mean(bg_patch[:, :, c]), np.std(bg_patch[:, :, c]) + 1e-5
        obj_patch[:, :, c] = (obj_patch[:, :, c] - obj_mean) / obj_std * bg_std + bg_mean
    return np.clip(obj_patch, 0, 255).astype(np.uint8)


def safe_imread(image_path):
    if not os.path.exists(image_path):
        return None
    return cv2.imread(image_path)


def get_car_surface_mask_dominant_color(img, color_thresh=30):
    """
    Generate a vehicle-surface mask using the dominant color in LAB space.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab_img], [0, 1, 2], None, [32] * 3, [0, 256] * 3)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)

    bin_size = 256 / 32
    dominant_lab = np.array([(i + 0.5) * bin_size for i in max_idx], dtype=np.float32)

    diff = np.linalg.norm(lab_img.astype(np.float32) - dominant_lab, axis=2)
    mask = (diff < color_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def get_random_point_in_mask(mask, obj_w, obj_h, trials=2000):
    """
    Sample a random top-left point where the full object region is inside the valid mask.
    """
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return None

    h_mask, w_mask = mask.shape[:2]

    for _ in range(trials):
        x = random.randint(0, max(0, w_mask - obj_w))
        y = random.randint(0, max(0, h_mask - obj_h))
        region = mask[y:y + obj_h, x:x + obj_w]
        if region.shape == (obj_h, obj_w) and np.all(region == 255):
            return x, y

    return None


def restore_highlight(original_patch, patched_patch, blend_ratio=0.01, x1=0, y1=0):
    h, w = original_patch.shape[:2]
    patched_crop = patched_patch[y1:y1 + h, x1:x1 + w]

    ori_gray = cv2.cvtColor(original_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pat_gray = cv2.cvtColor(patched_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = (ori_gray - pat_gray) * blend_ratio

    restored_crop = patched_crop.astype(np.float32)
    for c in range(3):
        restored_crop[..., c] = np.clip(restored_crop[..., c] + diff, 0, 255)
    restored_crop = restored_crop.astype(np.uint8)

    patched_patch[y1:y1 + h, x1:x1 + w] = restored_crop
    return patched_patch


def adjust_color_and_blend(clean_img, obj_crop, x1, y1,
                           feather_ratio=0.01,
                           color_blend_ratio=0.5,
                           seamless_alpha=0.15):
    h, w = obj_crop.shape[:2]
    original_patch = clean_img[y1:y1 + h, x1:x1 + w]

    # Color / texture alignment
    color_matched = match_histograms(obj_crop, original_patch, channel_axis=-1).astype(np.float32)
    texture_matched = match_texture_statistics(obj_crop.copy(), original_patch).astype(np.float32)
    matched = ((1 - color_blend_ratio) * texture_matched + color_blend_ratio * color_matched).astype(np.uint8)

    # Alpha blending
    blended_obj = cv2.addWeighted(
        matched.astype(np.float32), seamless_alpha,
        obj_crop.astype(np.float32), 1 - seamless_alpha, 0
    )
    blended_obj = np.clip(blended_obj, 0, 255).astype(np.uint8)

    # Feather mask
    mask = np.ones((h, w), dtype=np.uint8) * 255
    blur_size = max(3, int(min(w, h) * feather_ratio * 100) | 1)
    blurred_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    center = (x1 + w // 2, y1 + h // 2)

    # Seamless clone
    patched_patch = cv2.seamlessClone(blended_obj, clean_img.copy(), blurred_mask, center, cv2.MIXED_CLONE)

    # Restore highlights
    restored = restore_highlight(original_patch, patched_patch, blend_ratio=0.01, x1=x1, y1=y1)
    return restored


def write_yolo_label(label_path, x1, y1, w_obj, h_obj, img_w, img_h, cls_id=1):
    cx = x1 + w_obj / 2
    cy = y1 + h_obj / 2
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {cx / img_w:.6f} {cy / img_h:.6f} {w_obj / img_w:.6f} {h_obj / img_h:.6f}\n")


def make_divisible_by_8(value):
    return max((value // 8) * 8, 8)


# ===== Placement helper =====
def try_place_on_bg(clean_img, damage_img, car_mask):
    """
    Try placing the damage image from larger scales to smaller scales.
    """
    for i, scale in enumerate(SCALES):
        new_w = make_divisible_by_8(int(damage_img.shape[1] * scale))
        new_h = make_divisible_by_8(int(damage_img.shape[0] * scale))
        if new_w < 8 or new_h < 8:
            continue

        resized = cv2.resize(damage_img, (new_w, new_h))
        trials = TRIALS_FIRST_SCALE if i == 0 else TRIALS_OTHER_SCALES
        pt = get_random_point_in_mask(car_mask, new_w, new_h, trials=trials)
        if pt:
            x1, y1 = pt
            return resized, x1, y1

    return None, None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Composite refined dent images onto clean vehicle backgrounds and generate YOLO labels."
    )

    parser.add_argument(
        "--damage_images_dir",
        type=str,
        required=True,
        help="Directory containing clean background vehicle images used for overlay."
    )
    parser.add_argument(
        "--damage_labels_dir",
        type=str,
        required=True,
        help="Directory containing label files for the background images. It is used to identify images without existing damage annotations."
    )
    parser.add_argument(
        "--damage_defect_folder",
        type=str,
        required=True,
        help="Directory containing refined dent images generated by the refinement stage."
    )
    parser.add_argument(
        "--new_damage_img_dir",
        type=str,
        required=True,
        help="Output directory where the final composited images will be saved."
    )
    parser.add_argument(
        "--new_damage_label_dir",
        type=str,
        required=True,
        help="Output directory where the generated YOLO-format label files will be saved."
    )
    parser.add_argument(
        "--visual_output_dir",
        type=str,
        required=True,
        help="Output directory where visualization images with bounding boxes will be saved."
    )

    return parser.parse_args()


# ===== Main =====
def main():
    args = parse_args()

    damage_images_dir = args.damage_images_dir
    damage_labels_dir = args.damage_labels_dir
    damage_ref_folder = args.damage_ref_folder
    new_damage_img_dir = args.new_damage_img_dir
    new_damage_label_dir = args.new_damage_label_dir
    visual_output_dir = args.visual_output_dir

    os.makedirs(new_damage_img_dir, exist_ok=True)
    os.makedirs(new_damage_label_dir, exist_ok=True)
    os.makedirs(visual_output_dir, exist_ok=True)

    used_ref_images = set()
    ref_files = [
        f for f in os.listdir(damage_ref_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # Background images without existing damage labels
    damage_list = []
    for f in os.listdir(damage_images_dir):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            base_name, _ = os.path.splitext(f)
            label_path = os.path.join(damage_labels_dir, base_name + ".txt")
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                damage_list.append(f)

    if not damage_list:
        print("No clean background images were found.")
        return

    while len(used_ref_images) < len(ref_files):
        # Select each refined dent image once
        ref_img_name = random.choice(list(set(ref_files) - used_ref_images))
        used_ref_images.add(ref_img_name)

        ref_img_path = os.path.join(damage_ref_folder, ref_img_name)
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            print(f"Failed to load reference image: {ref_img_name}")
            continue

        success = False
        for _ in range(MAX_BG_TRIES_PER_REF):
            bg_img_name = random.choice(damage_list)
            base_name, _ = os.path.splitext(bg_img_name)

            clean_img = safe_imread(os.path.join(damage_images_dir, bg_img_name))
            if clean_img is None:
                continue

            car_mask = get_car_surface_mask_dominant_color(clean_img)
            final_obj, x1, y1 = try_place_on_bg(clean_img, ref_img, car_mask)

            if final_obj is None:
                continue

            # Composite
            final_img = adjust_color_and_blend(clean_img.copy(), final_obj, x1, y1)

            # Save image
            stem = os.path.splitext(ref_img_name)[0]
            new_img_name = f"{base_name}_{stem}_augd.jpg"
            new_img_path = os.path.join(new_damage_img_dir, new_img_name)
            cv2.imwrite(new_img_path, final_img)

            # Save label
            new_label_name = f"{base_name}_{stem}_augd.txt"
            new_label_path = os.path.join(new_damage_label_dir, new_label_name)
            h_obj, w_obj = final_obj.shape[:2]
            img_h, img_w = clean_img.shape[:2]
            write_yolo_label(new_label_path, x1, y1, w_obj, h_obj, img_w, img_h)

            # Save visualization
            vis_img = final_img.copy()
            cv2.rectangle(vis_img, (x1, y1), (x1 + w_obj, y1 + h_obj), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                "damage",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            vis_img_path = os.path.join(visual_output_dir, new_img_name)
            cv2.imwrite(vis_img_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

            print(f"Saved: {new_img_name}, label: {new_label_name}")
            success = True
            break

        if not success:
            print(f"Placement failed: {ref_img_name}")

    print("All refined dent images have been processed.")


if __name__ == "__main__":
    main()
