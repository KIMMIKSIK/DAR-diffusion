
import os
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler


# ──────────────────────────────
# 0) CMD 인자 설정
# ──────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DreamBooth fine-tuning for car dent images using <sks> token."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(os.getcwd(), "5test"),
        help="Path to training image folder. Default: ./5test"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained Stable Diffusion model"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Learning rate"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image resize size"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0005,
        help="Early stopping min delta"
    )

    return parser.parse_args()


# ──────────────────────────────
# 1) Dataset 정의
# ──────────────────────────────
class DamageDataset(Dataset):
    def __init__(self, image_dir, size=512):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in: {image_dir}")

        # 네가 올린 원본 코드와 동일하게 3개 프롬프트만 사용
        self.prompts = [
            "a close-up of a <sks> dent on the car surface",
            "a close-up of a car door with <sks> dent",
            "a <sks> dent on a vehicle surface",
        ]

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(img)
        prompt = random.choice(self.prompts)

        return {
            "pixel_values": pixel_values,
            "instance_prompt": prompt
        }


# ──────────────────────────────
# 2) EarlyStopping 클래스
# ──────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ──────────────────────────────
# 3) main
# ──────────────────────────────
def main():
    args = parse_args()

    # ──────────────────────────────
    # 파이프라인 로드 및 구성
    # ──────────────────────────────
    pretrained = args.pretrained

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(pretrained)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = pipe.to(device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # ──────────────────────────────
    # <sks> 토큰 추가 및 임베딩 초기화
    # ──────────────────────────────
    new_token = "<sks>"

    if new_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([new_token])
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))

        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data

        dent_id = tokenizer.convert_tokens_to_ids("dent")
        sks_id = tokenizer.convert_tokens_to_ids("<sks>")

        token_embeds[sks_id] = token_embeds[dent_id].clone()

        print("Added <sks> token and initialized it from 'dent' token.")
    else:
        print("<sks> token already exists.")

    # ──────────────────────────────
    # DDPM 스케줄러로 변경
    # ──────────────────────────────
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # ──────────────────────────────
    # VAE는 Freeze
    # ──────────────────────────────
    vae = pipe.vae
    unet = pipe.unet

    for p in vae.parameters():
        p.requires_grad = False

    # 학습 모드 명시
    text_encoder.train()
    unet.train()
    vae.eval()

    # ──────────────────────────────
    # 데이터 준비
    # ──────────────────────────────
    image_dir = args.image_dir

    dataset = DamageDataset(image_dir, size=args.size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    print(f"Total training images: {len(dataset)}")
    print(f"Image directory: {image_dir}")

    # ──────────────────────────────
    # Optimizer
    # ──────────────────────────────
    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()) + list(unet.parameters()),
        lr=args.lr
    )

    # ──────────────────────────────
    # 학습 루프
    # ──────────────────────────────
    num_epochs = args.num_epochs
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta
    )

    print("Start training...")

    for epoch in range(num_epochs):
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            prompts = batch["instance_prompt"]

            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            encoder_hidden_states = text_encoder(**text_inputs).last_hidden_state

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                pipe.scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()

            noisy_latents = pipe.scheduler.add_noise(
                latents,
                noise,
                timesteps
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)

        print(f"🔍 Epoch {epoch} Average Loss: {avg_loss:.4f}")

        early_stopper(avg_loss)

        print(
            f"Best Loss: {early_stopper.best_loss:.6f}, "
            f"Counter: {early_stopper.counter}/{early_stopper.patience}"
        )

        if early_stopper.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break

    # ──────────────────────────────
    # 모델 저장
    # ──────────────────────────────
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    pipe.save_pretrained(save_dir)

    print("✅ Training complete. Model saved to", save_dir)


if __name__ == "__main__":
    main()
