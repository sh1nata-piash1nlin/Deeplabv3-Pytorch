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
    p.add_argument("-t", "--test_path", type=str, default="../image_testing/airplane1.jpg")
    p.add_argument("-c", "--checkpoint_path", type=str, default="trained_model/bestVOC.pt")
    p.add_argument("-s", "--image_size", type=int, default=224)
    return p.parse_args()


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'deeplabv3_mobilenet_v3_large',
        weights=None
    )
    model.to(device).eval()
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    # remove aux_classifier params if present
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
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # (1, C, H, W)

    # Inference
    with torch.no_grad():
        out = model(input_tensor)['out'][0]    # [C, H, W]
        output_predictions = out.argmax(0)     # [H, W]

    orig_np = np.array(img.resize((args.image_size, args.image_size)))

    num_classes = 21
    # Create base multipliers
    palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
    # Generate colors: (num_classes, 3)
    colors = (torch.arange(num_classes, dtype=torch.int64)[:, None] * palette_base[None, :]) % 255
    colors = colors.numpy().astype("uint8")

    # Create a paletted image from the prediction mask
    pal_img = Image.fromarray(output_predictions.byte().cpu().numpy(), mode="P")
    pal_img.putpalette(colors.flatten())
    rgb_img = pal_img.convert("RGB")

    # Plot original and segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#333333')
    for ax in axes:
        ax.set_facecolor('#333333')
        ax.axis('off')

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", color='white')

    axes[1].imshow(rgb_img)
    axes[1].set_title("Predicted Segmentation", color='white')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    test(args)
