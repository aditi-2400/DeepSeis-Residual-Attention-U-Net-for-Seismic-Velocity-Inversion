"""
train.py - Training script for Residual Attention UNet.
"""
import torch
import torch.optim as optim
from tqdm import tqdm
from model import ResAttentionUNet
from utils import calculate_mape

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResAttentionUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,     # reduce LR by half
    patience=3,     # epochs to wait before reducing
    verbose=True
)
loss_fn = nn.MSELoss()

def train_res_attention_unet(train_loader, val_loader, model, optimizer, loss_fn, scheduler,
                         max_epochs=100, patience=10, checkpoint_path="checkpoint_resattunet.pth"):
    device = next(model.parameters()).device

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mape = checkpoint['best_val_mape']
        print(f"Resuming from epoch {start_epoch} with best_val_mape {best_val_mape:.6f}")
    else:
        start_epoch = 0
        best_val_mape = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss = 0.0
        for seismic, velocity in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            seismic, velocity = seismic.to(device), velocity.to(device)
            optimizer.zero_grad()
            output = model(seismic)
            output = nn.functional.interpolate(output, size=velocity.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(output, velocity)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        mapes = []
        with torch.no_grad():
            for seismic, velocity in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                seismic, velocity = seismic.to(device), velocity.to(device)
                output = model(seismic)
                output = nn.functional.interpolate(output, size=velocity.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(output, velocity)
                val_loss += loss.item()
                mape = calculate_mape(velocity.cpu().numpy(), output.cpu().numpy())
                mapes.append(mape)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = np.mean(mapes)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}, Validation MAPE = {avg_val_mape:.4f}")
        scheduler.step(avg_val_mape)
        # Early stopping & checkpoint
        if avg_val_mape < best_val_mape:
            best_val_mape = avg_val_mape
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mape': best_val_mape
            }, checkpoint_path)
            print("--> New Best Model Saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience {patience})")
                break

    print(f"Training complete. Best Validation MAPE: {best_val_mape:.6f}")