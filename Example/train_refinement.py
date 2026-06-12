import os
import random
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image-conditioned dent refinement with StableDiffusionImg2ImgPipeline"
    )

    # Required paths
    parser.add_argument("--model_dir", type=str, required=True, help="Path to local pretrained model directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save refined images")
    parser.add_argument(
        "--reference_crop_dir",
        type=str,
        required=True,
        help="Path to reference dent crop directory for size range estimation",
    )

    # Optional settings
    parser.add_argument("--img2img_ratio", type=float, default=0.75, help="Ratio of images to refine")
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="If set, non-refined or failed images are saved as original copies",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--max_attempts", type=int, default=3, help="Maximum retry attempts per image")
    parser.add_argument("--extent_threshold", type=float, default=0.994, help="Contour extent threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ✅ 환경 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ✅ 출력 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # ✅ 파이프라인 로드
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype
    ).to(device)

    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    # ✅ 프롬프트
    prompts = [
        "a car surface with <sks> dent, close-up view, metallic reflection, realistic detail",
        "a close-up of a <sks> dent on the car surface",
        "a car surface with <sks> dent, close-up view, realistic detail",
    ]
    random.shuffle(prompts)

    # ✅ reference 크기 기반 리사이즈 범위 계산
    widths, heights = [], []
    for fn in os.listdir(args.reference_crop_dir):
        if fn.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                img = Image.open(os.path.join(args.reference_crop_dir, fn))
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except Exception:
                print(f"⚠️ Invalid reference image skipped: {fn}")

    if not widths or not heights:
        raise ValueError("❌ No valid reference images found.")

    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)

    # ✅ 입력 이미지 목록
    file_list = [
        fn for fn in os.listdir(args.input_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    random.shuffle(file_list)

    if len(file_list) == 0:
        raise ValueError("❌ No valid input images found.")

    num_to_transform = int(len(file_list) * args.img2img_ratio)
    transform_targets = set(random.sample(file_list, num_to_transform))

    image_count = 1
    num_transformed = 0
    num_failed = 0
    num_original = 0

    # ✅ 생성 루프
    for fn in file_list:
        src_path = os.path.join(args.input_dir, fn)
        dst_path = os.path.join(args.output_dir, f"car_dents_{image_count:04d}.jpg")

        try:
            orig = Image.open(src_path).convert("RGB")
        except Exception:
            print(f"❌ Failed to open image: {fn}")
            continue

        # 변형 대상이 아니면 원본 저장
        if fn not in transform_targets:
            if args.save_original:
                orig.save(dst_path, quality=100)
                print(f"📂 Original saved: {dst_path}")
                image_count += 1
                num_original += 1
            continue

        crop_w = random.randint(min_w, min(max_w, 900))
        crop_h = random.randint(min_h, min(max_h, 900))
        if crop_w >= 900 or crop_h >= 900:
            crop_w //= 2
            crop_h //= 2

        resized = orig.resize((crop_w, crop_h))
        nw = ((crop_w + 63) // 64) * 64
        nh = ((crop_h + 63) // 64) * 64

        cond = Image.new("RGB", (nw, nh), (255, 255, 255))
        cond.paste(resized, (0, 0))

        strength = round(random.uniform(0.2, 0.4), 2)
        guidance_scale = round(random.uniform(3.0, 5.0), 2)
        full_prompt = random.choice(prompts)

        attempt = 0
        saved = False
        base_seed = random.randint(0, 99999)

        while attempt < args.max_attempts:
            try:
                result = pipe(
                    prompt=full_prompt,
                    image=cond,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(base_seed + attempt),
                ).images[0]
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed for {fn}: {e}")
                attempt += 1
                continue

            result_cropped = result.crop((0, 0, crop_w, crop_h))
            gray = np.array(result_cropped.convert("L"))
            _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                attempt += 1
                continue

            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            extent = area / (w * h) if w * h else 0

            if extent > args.extent_threshold:
                result_cropped.save(dst_path, quality=100)
                print(f"✅ Refined image saved: {dst_path} (extent={extent:.4f})")
                image_count += 1
                num_transformed += 1
                saved = True
                break
            else:
                attempt += 1

        if not saved:
            print(f"⚠️ {fn}: refinement failed after max attempts")
            if args.save_original:
                orig.save(dst_path, quality=100)
                print(f"📂 Fallback original saved: {dst_path}")
                image_count += 1
                num_original += 1
            num_failed += 1

    # ✅ 요약 출력
    print("\n🎯 Processing completed")
    print(f"📊 Original images saved: {num_original}")
    print(f"📊 Transformation targets: {len(transform_targets)}")
    print(f"📊 Successfully refined: {num_transformed}")
    print(f"📊 Failed refinements: {num_failed}")
    print(f"✅ Success rate over all images: {num_transformed / len(file_list) * 100:.2f}%")
    if len(transform_targets) > 0:
        print(f"✅ Success rate over selected targets: {num_transformed / len(transform_targets) * 100:.2f}%")


if __name__ == "__main__":
    main()

    
