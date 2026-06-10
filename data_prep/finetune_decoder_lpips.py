import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= CONFIGURATION =================
# Input and Output Paths
VAE_CKPT = "./checkpoints/vae_best.pt"
CROPS_DIR = "./data/crops"
OUT_CKPT = "./checkpoints/decoder_ft_lpips.pt"
VIZ_DIR  = "./outputs/decoder_ft_viz"

# Data Normalization Settings
# Set to "01" if original images are [0,1], "11" if [-1,1]
IN_RANGE = "01" 

# Hyperparameters for Sharpness
W_L1 = 0.1       # Reduced L1 weight to prevent over-smoothing
W_LP = 1.5       # Increased LPIPS weight to force structural detail
LR = 1e-4        # Learning rate
EPOCHS = 10      # Number of fine-tuning epochs
BATCH_SIZE = 32
# =================================================

class CropDataset(Dataset):
    def __init__(self, root, img_size=96, limit=None, in_range="01"):
        import torchvision.transforms as T
        from PIL import Image  # noqa
        self.in_range = in_range
        exts = ["png", "jpg", "jpeg", "tif", "tiff", "PNG", "JPG", "JPEG", "TIF", "TIFF"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        paths = sorted(paths)
        if limit is not None:
            paths = paths[:limit]
        if len(paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.paths = paths
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        x = self.tf(Image.open(self.paths[i]).convert("RGB"))
        if self.in_range == "11":
            x = x * 2 - 1
        return x

@torch.no_grad()
def save_pair(x, xhat, out_png):
    import torchvision.utils as vutils
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if IN_RANGE == "11":
        x_vis = ((x + 1) / 2).clamp(0, 1)
        xh_vis = ((xhat + 1) / 2).clamp(0, 1)
    else:
        x_vis = x.clamp(0, 1)
        xh_vis = xhat.clamp(0, 1)
    
    # Grid: Top row original, Bottom row reconstructed
    comparison = torch.cat([x_vis[:8], xh_vis[:8]], 0)
    grid = vutils.make_grid(comparison, nrow=8, padding=2)
    vutils.save_image(grid, out_png)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune VAE decoder with LPIPS.")
    ap.add_argument("--vae_ckpt", default="./checkpoints/vae_best.pt")
    ap.add_argument("--crops_dir", default="./data/crops")
    ap.add_argument("--out_ckpt", default="./checkpoints/decoder_ft_lpips.pt")
    ap.add_argument("--viz_dir", default="./outputs/decoder_ft_viz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--in_range", choices=["01", "11"], default="01")
    ap.add_argument("--w_l1", type=float, default=0.1)
    ap.add_argument("--w_lpips", type=float, default=1.5)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    import lpips  # heavy/optional: imported after --help works
    VAE_CKPT, CROPS_DIR, OUT_CKPT, VIZ_DIR = args.vae_ckpt, args.crops_dir, args.out_ckpt, args.viz_dir
    LR, EPOCHS, BATCH_SIZE = args.lr, args.epochs, args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    IN_RANGE, W_L1, W_LP = args.in_range, args.w_l1, args.w_lpips
    
    # 1. Load Checkpoint and Model
    # Assuming VAE class is imported from your training script
    try:
        from .train_vae_full import VAE
    except ImportError:
        from train_vae_full import VAE 
    
    checkpoint = torch.load(VAE_CKPT, map_location="cpu")
    zdim = int(checkpoint["zdim"])
    img_size = int(checkpoint.get("img_size", 96))

    vae = VAE(zdim=zdim, img_size=img_size).to(device)
    vae.load_state_dict(checkpoint["model"], strict=True)
    
    # 2. Freeze Encoder, Enable Decoder Training
    vae.train()
    for p in vae.enc.parameters():
        p.requires_grad = False
    for p in vae.dec.parameters():
        p.requires_grad = True

    # 3. Setup Optimizer and Perceptual Loss
    optimizer = torch.optim.Adam(vae.dec.parameters(), lr=LR)
    criterion_l1 = nn.L1Loss()
    criterion_lpips = lpips.LPIPS(net="vgg").to(device).eval()
    for p in criterion_lpips.parameters():
        p.requires_grad = False

    # 4. Data Loading
    dataset = CropDataset(CROPS_DIR, img_size=img_size, in_range=args.in_range)
    if len(dataset) < args.batch_size:
        raise ValueError(f"Dataset has {len(dataset)} images but batch_size={args.batch_size} with drop_last=True.")
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )

    print(f"Starting Fine-tuning: W_L1={W_L1}, W_LP={W_LP}, LR={LR}")

    global_step = 0
    for epoch in range(EPOCHS):
        for x in dataloader:
            x = x.to(device, non_blocking=True)

            # Reconstruct using mu for stability
            with torch.no_grad():
                mu, _ = vae.enc(x)
                z = mu

            xhat = vae.dec(z)

            # Ensure strict range mapping for LPIPS [-1, 1]
            if IN_RANGE == "01":
                x_lpips = (x * 2 - 1).clamp(-1, 1)
                xhat_lpips = (xhat.clamp(0, 1) * 2 - 1).clamp(-1, 1)
                l1_term = criterion_l1(xhat.clamp(0, 1), x)
            else:
                x_lpips = x.clamp(-1, 1)
                xhat_lpips = xhat.clamp(-1, 1)
                l1_term = criterion_l1(xhat_lpips, x)

            # Combined Loss
            perceptual_term = criterion_lpips(xhat_lpips, x_lpips).mean()
            loss = (W_L1 * l1_term) + (W_LP * perceptual_term)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step {global_step} | Total Loss: {loss.item():.4f} | LPIPS: {perceptual_term.item():.4f}")
            
            if global_step % 500 == 0:
                save_pair(
                    x.detach().cpu(), 
                    xhat.detach().cpu(), 
                    f"{VIZ_DIR}/epoch{epoch}_step{global_step}.png"
                )
            
            global_step += 1

    # 5. Save Enhanced Decoder Checkpoint
    final_output = dict(checkpoint)
    final_output["model"] = vae.state_dict()
    final_output["fine_tuned"] = True
    os.makedirs(os.path.dirname(OUT_CKPT) or '.', exist_ok=True)
    torch.save(final_output, OUT_CKPT)
    print(f"Fine-tuned model saved to: {OUT_CKPT}")

if __name__ == "__main__":
    main()