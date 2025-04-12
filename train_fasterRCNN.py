"""
@author: Nguyen Duc "sh1nata" Tri <tri14102004@gmail.com>
"""

import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from src.VOCDataBuild import VOCDataset
import torch
from src.utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import shutil

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = Resize((args.image_size, args.image_size))

    train_data= VOCDataset(root=args.dataPath, year=args.year, image_set="train", download=False, transform=transform, target_transform=target_transform)
    valid_data = VOCDataset(root=args.dataPath, year=args.year, image_set="val", download=False, transform=transform, target_transform=target_transform)
    train_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "drop_last": False,
    }
    valid_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "drop_last": False,
    }
    train_dataloader = DataLoader(train_data, **train_params)
    valid_dataloader = DataLoader(valid_data, **valid_params)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None
    acc_metric = MulticlassAccuracy(num_classes=len(train_data.classes)).to(device)
    mIOU_metric = MulticlassJaccardIndex(num_classes=len(train_data.classes)).to(device)
    start_epoch = 0
    total_iters = len(train_dataloader)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=args.momentum)
        scheduler = MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=args.gamma)

    if args.continue_cp and os.path.isfile(args.continue_cp):     #continue from this cp if stop training suddenly
        checkpoint = torch.load(args.continue_cp,
                                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        best_mIOU = checkpoint["best_mIOU"]
    else:
        start_epoch = 0
        best_acc = -1
        best_mIOU = -1

    if os.path.isdir(args.log_folder): # save tensorboard
        shutil.rmtree(args.log_folder)
    os.mkdir(args.log_folder)
    if not os.path.isdir(args.cp_folder): #save checkpoints
        os.mkdir(args.cp_folder)
    writer = SummaryWriter(args.log_folder)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        all_train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            result = model(images)
            pred = result['out']
            loss = criterion(pred, labels)
            all_train_loss.append(loss.item())
            avg_loss = np.mean(all_train_loss)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, avg_loss))
            writer.add_scalar("Train/Loss", avg_loss, epoch * total_iters + iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        #progress_bar = tqdm(valid_dataloader, colour="green")
        all_val_loss = []
        val_acc = []
        val_mIOU = []
        with torch.inference_mode():
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                result = model(images)
                pred = result['out'] # # B, C, H, W  (pred).  B, H, W (gt)
                loss = criterion(pred, labels)
                all_val_loss.append(loss.item())
                val_acc.append(acc_metric(pred, labels).item())
                val_mIOU.append(mIOU_metric(pred, labels).item())
                #progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, np.mean(loss.item())))

        avg_loss = np.mean(all_val_loss)
        avg_acc = np.mean(val_acc)
        avg_mIOU = np.mean(val_mIOU)
        #print("Accuracy: {:0.4f}. mIOU: {:0.4f}".format(avg_acc, avg_mIOU))
        writer.add_scalar("Validation/Loss", avg_loss, epoch)
        writer.add_scalar("Validation/Accuracy", avg_acc, epoch)
        writer.add_scalar("Validation/mIOU", avg_mIOU, epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "best_mIOU": best_mIOU,
            # "batch_size": args.batch_size,
        }
        torch.save(checkpoint, os.path.join(args.cp_folder, "lastVOC.pt"))
        if avg_acc > best_acc and avg_mIOU > best_mIOU:
            best_mIOU = avg_mIOU
            best_acc = avg_acc
            torch.save(checkpoint, os.path.join(args.cp_folder, "bestVOC.pt"))

if __name__ == '__main__':
    train(get_args())
