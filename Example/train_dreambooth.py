
# <sks>토큰 사용 #이거쓰면 됩니다!!!!!

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler

# ──────────────────────────────
# 1) Dataset 정의
# ──────────────────────────────
class DamageDataset(Dataset):
    def __init__(self, image_dir, size=512):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", "jpeg", ".png"))
        ]
        self.prompts = [
            "a close-up of a <sks> dent on the car surface",
            "a <sks> damage with car surface",
            "a close-up of a car door with <sks> dent",
            "a <sks> dent on a vehicle surface",
           
        ]
    
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
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
    def __init__(self, patience=3, min_delta=0.001):
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
# 3) 파이프라인 로드 및 구성
# ──────────────────────────────
pretrained = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(pretrained)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

# ✅ <sks> 토큰 추가 및 임베딩 초기화
new_token = "<sks>"
if new_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([new_token])
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
    dent_id = tokenizer.convert_tokens_to_ids("dent")
    sks_id = tokenizer.convert_tokens_to_ids("<sks>")
    token_embeds[sks_id] = token_embeds[dent_id].clone()

# ✅ DDPM 스케줄러로 변경
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# ✅ VAE는 Freeze
vae = pipe.vae
unet = pipe.unet
for p in vae.parameters():
    p.requires_grad = False

# ──────────────────────────────
# 4) 데이터 준비
# ──────────────────────────────

image_dir = r"C:\Users\msik\python_msik\yolo_car\Custom_path"
dataset = DamageDataset(image_dir, size=512)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ──────────────────────────────
# 5) Optimizer
# ──────────────────────────────
optimizer = torch.optim.AdamW(
    list(text_encoder.parameters()) + list(unet.parameters()),
    lr=2e-6
)

# ──────────────────────────────
# 6) 학습 루프
# ──────────────────────────────
num_epochs = 100
early_stopper = EarlyStopping(patience=5, min_delta=0.0005)

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
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
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

    if early_stopper.early_stop:
        print(f"🛑 Early stopping triggered at epoch {epoch}")
        break

# ──────────────────────────────
# 7) 모델 저장
# ──────────────────────────────
save_dir = "./Dreambooth_test5pce_weight"
os.makedirs(save_dir, exist_ok=True)
pipe.save_pretrained(save_dir)
print("✅ Training complete. Model saved to", save_dir)

