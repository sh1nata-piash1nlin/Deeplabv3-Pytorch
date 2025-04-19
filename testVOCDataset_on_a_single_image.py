"""
@author: Nguyen Duc "sh1nata" Tri <tri14102004@gmail.com>
"""

import argparse
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'deeplabv3_mobilenet_v3_large',
        weights=None
    )
    model.to(device).eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier')}
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
    orig_np = np.array(img.resize((args.image_size, args.image_size)))

    # 1) Build a ListedColormap where class 0 is dark
    num_classes = 21
    base = plt.get_cmap("tab20", num_classes)
    palette = [base(i) for i in range(num_classes)]
    palette[0] = (0.05, 0.05, 0.05, 1.0)     # almost black for background
    cmap = ListedColormap(palette)

    # 2) Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Optional: Make the canvas itself dark
    fig.patch.set_facecolor('#333333')
    axes[0].set_facecolor('#333333')
    axes[1].set_facecolor('#333333')

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", color='white')
    axes[0].axis("off")

    im = axes[1].imshow(
        pred,
        cmap=cmap,
        vmin=0,
        vmax=num_classes-1,
        interpolation='nearest'
    )
    axes[1].set_title("Predicted Segmentation", color='white')
    axes[1].axis("off")

    # colorbar
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    cbar = fig.colorbar(
        im,
        ax=axes[1],
        ticks=np.arange(num_classes),
        fraction=0.046,
        pad=0.04
    )
    cbar.ax.set_yticklabels(classes, color='white')
    cbar.outline.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = get_args()
    test(args)
