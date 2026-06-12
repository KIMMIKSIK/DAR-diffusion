import os
import glob
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative few-shot bbox crops using ResNet-50 features, K-means clustering, and diversity-aware search."
    )

    parser.add_argument("--images_dir", type=str, required=True, help="Path to training damage images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to YOLO-format label files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save selected bbox crops and PCA plot")

    parser.add_argument("--k", type=int, default=5, help="Number of representative bbox crops to select")
    parser.add_argument("--sim_threshold", type=float, default=0.98, help="Cosine similarity threshold for redundancy filtering")
    parser.add_argument("--top_m_per_cluster", type=int, default=5, help="Top-M candidate samples per cluster")
    parser.add_argument("--n_comb_trials", type=int, default=3000, help="Number of candidate combination trials")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


# -----------------------------
# 유틸 함수: YOLO bbox → 픽셀 좌표
# -----------------------------
def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = max(0, x_c - w / 2)
    y1 = max(0, y_c - h / 2)
    x2 = min(img_w, x_c + w / 2)
    y2 = min(img_h, y_c + h / 2)

    return int(x1), int(y1), int(x2), int(y2)


# -----------------------------
# 1. Feature extractor 준비
# -----------------------------
def build_feature_extractor(device):
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(pretrained=True)

    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    feature_extractor.to(device)

    return feature_extractor


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# 2. bbox crop별 prototype 추출
# -----------------------------
def extract_bbox_prototypes(images_dir, labels_dir, device):
    feature_extractor = build_feature_extractor(device)

    image_paths = sorted(
        list(glob.glob(os.path.join(images_dir, "*.jpg"))) +
        list(glob.glob(os.path.join(images_dir, "*.jpeg"))) +
        list(glob.glob(os.path.join(images_dir, "*.png"))) +
        list(glob.glob(os.path.join(images_dir, "*.bmp")))
    )

    prototypes = []
    crop_infos = []

    for img_path in image_paths:
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, img_name + ".txt")

        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            continue

        for bbox_idx, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id, x_c, y_c, w, h = parts

            try:
                x_c, y_c, w, h = map(float, (x_c, y_c, w, h))
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            # 핵심: bbox crop 하나를 하나의 샘플로 사용
            patch = img.crop((x1, y1, x2, y2))
            patch_tensor = transform(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = feature_extractor(patch_tensor)
                feat = feat.view(-1)

            prototypes.append(feat.cpu())

            crop_infos.append({
                "img_path": img_path,
                "img_name": img_name,
                "class_id": class_id,
                "bbox_idx": bbox_idx,
                "xyxy": (x1, y1, x2, y2),
            })

    if len(prototypes) == 0:
        raise RuntimeError("No valid bbox crops were found.")

    prototypes = torch.stack(prototypes, dim=0)
    prototypes = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)

    return crop_infos, prototypes


# -----------------------------
# 3. 평가 함수
# -----------------------------
def average_pairwise_cosine_distance(features: torch.Tensor, indices):
    if len(indices) <= 1:
        return 0.0

    feat_sel = features[indices]
    sim = feat_sel @ feat_sel.t()
    dist = 1.0 - sim

    k = len(indices)
    upper_sum = dist.triu(diagonal=1).sum().item()
    denom = k * (k - 1) / 2

    return upper_sum / denom


# -----------------------------
# 4. K-means 기반 후보군 구성
# -----------------------------
def build_cluster_candidate_pools(prototypes: torch.Tensor, k: int, top_m: int, random_seed: int):
    feats_np = prototypes.cpu().numpy()
    n_clusters = min(k, feats_np.shape[0])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,
    )

    cluster_labels = kmeans.fit_predict(feats_np)
    centers = kmeans.cluster_centers_

    cluster_candidate_pools = []

    for c in range(n_clusters):
        cluster_idx = np.where(cluster_labels == c)[0]

        if len(cluster_idx) == 0:
            continue

        cluster_feats = feats_np[cluster_idx]
        center = centers[c][None, :]

        dists = np.linalg.norm(cluster_feats - center, axis=1)
        sorted_local = np.argsort(dists)

        top_local = sorted_local[: min(top_m, len(sorted_local))]
        top_global = cluster_idx[top_local].tolist()

        cluster_candidate_pools.append(top_global)

    return cluster_candidate_pools


# -----------------------------
# 5. 대표 bbox crop 선택
# -----------------------------
def select_representative_samples(
    sample_infos,
    prototypes,
    k=5,
    sim_threshold=0.98,
    top_m=5,
    n_trials=3000,
    random_seed=42,
):
    N = prototypes.size(0)

    if N <= k:
        all_idx = list(range(N))
        return all_idx, [sample_infos[i] for i in all_idx]

    random.seed(random_seed)
    np.random.seed(random_seed)

    candidate_pools = build_cluster_candidate_pools(
        prototypes=prototypes,
        k=k,
        top_m=top_m,
        random_seed=random_seed,
    )

    if len(candidate_pools) < k:
        used = set()

        for pool in candidate_pools:
            used.update(pool)

        for idx in range(N):
            if idx not in used:
                candidate_pools.append([idx])

            if len(candidate_pools) == k:
                break

    sim_matrix = prototypes @ prototypes.t()

    def is_valid_set(indices):
        if len(indices) != len(set(indices)):
            return False

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if sim_matrix[indices[i], indices[j]].item() > sim_threshold:
                    return False

        return True

    def greedy_fallback():
        selected = []

        for pool in candidate_pools[:k]:
            picked = None

            for cand in pool:
                okay = True

                for prev in selected:
                    if sim_matrix[cand, prev].item() > sim_threshold:
                        okay = False
                        break

                if okay:
                    picked = cand
                    break

            if picked is None:
                picked = pool[0]

            selected.append(picked)

        return selected

    best_indices = greedy_fallback()
    best_score = average_pairwise_cosine_distance(prototypes, best_indices)

    for _ in range(n_trials):
        sampled = []

        for pool in candidate_pools[:k]:
            sampled.append(random.choice(pool))

        if not is_valid_set(sampled):
            continue

        score = average_pairwise_cosine_distance(prototypes, sampled)

        if score > best_score:
            best_score = score
            best_indices = sampled

    selected_infos = [sample_infos[i] for i in best_indices]

    print("\n[INFO] Final selected bbox crop indices:", best_indices)
    print(f"[INFO] Best average pairwise cosine distance: {best_score:.6f}")

    return best_indices, selected_infos


# -----------------------------
# 6. PCA 시각화
# -----------------------------
def visualize_pca(prototypes, selected_indices, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    N = prototypes.size(0)

    if N < 2:
        print("At least two bbox crops are required for PCA visualization. Skipping.")
        return

    feats = prototypes.cpu().numpy()

    print("Calculating PCA(2D)...")

    pca = PCA(n_components=2, random_state=0)
    feats_2d = pca.fit_transform(feats)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        feats_2d[:, 0],
        feats_2d[:, 1],
        s=10,
        alpha=0.3,
        label="All damage bbox crops",
    )

    sel_idx = np.array(selected_indices, dtype=int)

    plt.scatter(
        feats_2d[sel_idx, 0],
        feats_2d[sel_idx, 1],
        s=80,
        alpha=0.9,
        marker="*",
        label="Selected representative bbox crops",
    )

    for rank, idx in enumerate(sel_idx, start=1):
        x, y = feats_2d[idx]
        plt.text(x + 0.02, y + 0.02, str(rank), fontsize=9)

    plt.title("PCA of damage bbox crop prototypes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "damage_bbox_representative_pca.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"PCA plot saved to {save_path}")


# -----------------------------
# 7. 선택된 bbox crop 5개를 5test 폴더에 저장
# -----------------------------
def save_selected_bbox_crops(selected_crop_infos, output_dir):
    # output_dir 안에 5test 폴더 생성
    crop_output_dir = os.path.join(output_dir, "5test")
    os.makedirs(crop_output_dir, exist_ok=True)

    for rank, info in enumerate(selected_crop_infos, start=1):
        img_path = info["img_path"]
        img_name = info["img_name"]
        class_id = info["class_id"]
        bbox_idx = info["bbox_idx"]
        x1, y1, x2, y2 = info["xyxy"]

        img = Image.open(img_path).convert("RGB")
        crop = img.crop((x1, y1, x2, y2))

        save_name = f"selected_{rank:02d}_{img_name}_bbox{bbox_idx}_cls{class_id}.jpg"
        save_path = os.path.join(crop_output_dir, save_name)

        crop.save(save_path, quality=95)

        print(f"Saved selected crop {rank}: {save_path}")

    print(f"\nA total of {len(selected_crop_infos)} representative bbox crops were saved to {crop_output_dir}.")


# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1) Extracting bbox crop prototypes...")
    crop_infos, prototypes = extract_bbox_prototypes(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        device=device,
    )

    print(f"Total valid bbox crops: {len(crop_infos)}")

    print("\n2) Running K-means + candidate replacement + diversity search...")
    selected_indices, selected_crops = select_representative_samples(
        sample_infos=crop_infos,
        prototypes=prototypes,
        k=args.k,
        sim_threshold=args.sim_threshold,
        top_m=args.top_m_per_cluster,
        n_trials=args.n_comb_trials,
        random_seed=args.random_seed,
    )

    print(f"\n=== Selected representative {args.k} bbox crops ===")

    for i, info in enumerate(selected_crops, start=1):
        print(
            f"{i}. image={info['img_path']}, "
            f"bbox_idx={info['bbox_idx']}, "
            f"class={info['class_id']}, "
            f"xyxy={info['xyxy']}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    out_txt = os.path.join(args.output_dir, f"selected_damage_representative_bbox_{args.k}.txt")

    with open(out_txt, "w", encoding="utf-8") as f:
        for info in selected_crops:
            f.write(
                f"{info['img_path']}\t"
                f"bbox_idx={info['bbox_idx']}\t"
                f"class={info['class_id']}\t"
                f"xyxy={info['xyxy']}\n"
            )

    print(f"\nSelected bbox crop list saved to {out_txt}")

    print("\n3) Saving selected bbox crops...")
    save_selected_bbox_crops(
        selected_crop_infos=selected_crops,
        output_dir=args.output_dir,
    )

    print("\n4) Generating PCA visualization...")
    visualize_pca(
        prototypes=prototypes,
        selected_indices=selected_indices,
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
