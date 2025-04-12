from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.dataVOC import VOCDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.utils import *
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
from pprint import pprint

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = ToTensor()
    train_dataset = VOCDataset(root=args.dataPath, year=args.year, image_set="train", download=False, transform=transform)
    val_dataset = VOCDataset(root=args.dataPath, year=args.year, image_set="val", download=False, transform=transform)
    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn" : collate_fn,
    }
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn" : collate_fn,
    }
    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **train_params)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(train_dataset.categories))
    #create optimizer
    optimizer = None
    scheduler = None
    start_epoch = 0

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=args.momentum)
        scheduler = MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=args.gamma)
    #continue from this cp
    if args.continue_cp:
        checkpoint = torch.load(args.continue_cp, map_location=lambda storage, loc: storage.cude(torch.cuda.current_device()))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["map"]
    else:
        start_epoch = 0
        best_map = -1

    model.to(device)

    if os.path.isdir(args.log_folder): # save tensorboard
        shutil.rmtree(args.log_folder)
    os.mkdir(args.log_folder)
    if not os.path.isdir(args.cp_folder): #save checkpoints
        os.mkdir(args.cp_folder)

    writer = SummaryWriter(args.log_folder)

    total_iters = len(train_dataloader)
    #training and validation process
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        train_loss = []  #smoothing the tensorboard visualization
        for iter, (images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            images = [image.to(device) for image in images]
            labels = [{"boxes":target["boxes"].to(device), "labels":target["labels"].to(device)} for target in labels]
            #forward
            losses = model(images, labels)
            final_losses = sum([loss for loss in losses.values()])
            #backward
            final_losses.backward()
            optimizer.step()
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.epochs, final_losses.item()))
            train_loss.append(final_losses.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), epoch*total_iters + iter)

        model.eval()
        progress_bar = tqdm(val_dataloader, color="blue")
        metric = MeanAveragePrecision(iou_type="bbox")
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu"),
                })
            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"],
                })
            metric.update(preds, targets)

        result = metric.compute()
        pprint(result)
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "map": result["map"],
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.cp_folder, "last.pt"))
        if result["map"] > best_map:
            best_map = result["map"]
            torch.save(checkpoint, os.path.join(args.cp_folder, "best.pt"))





if __name__ == '__main__':
    train(get_args())
