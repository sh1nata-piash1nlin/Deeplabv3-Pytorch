"""
@author: Nguyen Duc "sh1nata" Tri <tri14102004@gmail.com>
"""

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train Deeplabv3")
    parser.add_argument("--image-size", "-i", type=int, default=224)
    parser.add_argument("--batch_size", "-b", type=int, default=2, help="batch size")
    parser.add_argument("--num_workers", "-w", type=int, default=os.cpu_count())
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--dataPath", "-d", type=str, default="./data")
    parser.add_argument("--optimizer","-op", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", "-l", type=float, default=0.001)
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--gamma", "-g", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--cp_folder", "-c", type=str, default="trained_model", help="folder path to save cp")
    parser.add_argument("--log_folder", "-log", type=str, default="tensorboard", help="folder path to gen tensorboard")
    parser.add_argument("--continue-cp", "-con", type=str, default=None) #if none, train from scratch
    args = parser.parse_args()
    return args