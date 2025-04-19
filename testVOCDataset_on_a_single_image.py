import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from src.VOCDataBuild import VOCDataset
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--test_path", type=str, default="../image_testing/dog1.jpeg")
    p.add_argument("-c", "--checkpoint_path", type=str, default="trained_model/bestVOC.pt")
    p.add_argument("-s", "--image_size", type=int, default=224)
    return p.parse_args()

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model (no pretrained weights here)
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_mobilenet_v3_large',
                           weights=None)
    model.to(device).eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    state_dict = {k:v for k,v in state_dict.items() if not k.startswith('aux_classifier')}
    model.load_state_dict(state_dict)

    # Preprocess
    img = Image.open(args.test_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = model(input_tensor)['out'][0]    # [C, H, W]
        pred = out.argmax(0).cpu().numpy()     # [H, W] int mask

    # Prepare for plotting
    orig_np = img.resize((args.image_size, args.image_size))
    orig_np = np.array(orig_np)

    num_classes = 21
    cmap = plt.get_cmap("tab20", num_classes)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(
    pred,
    cmap=cmap,
    vmin=0,
    vmax=num_classes-1,
    interpolation='nearest'   # ← prevent smoothing
)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    # add colorbar with class indices
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(num_classes), fraction=0.046, pad=0.04)
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor']
    cbar.ax.set_yticklabels(classes)  # label ticks with class names

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = get_args()
    test(args)