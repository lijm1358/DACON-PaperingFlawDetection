import glob
import json
import os
import random
from datetime import datetime
from importlib import import_module

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_model_path(base_path):
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d-%H:%M:%S")
    model_path = os.path.join(base_path, dt_string)
    os.makedirs(model_path, exist_ok=False)
    return model_path


def main(config):
    # Wandb Initialization
    wandb.init(
        project="papering",
        config={
            "Model": config["model"],
            "Dataset": config["dataset"],
            "Optimizer": config["optimizer"]["type"],
            "Learning Rate": config["optimizer"]["args"]["lr"],
            "Scheduler": config["scheduler"]["type"],
            "Epoch": config["params"]["epochs"],
            "Batch Size": config["params"]["batch_size"],
        },
        name="Yang)" + config["model"] + config["optimizer"]["args"]["lr"]
    )

    seed_everything(config["seed"])
    all_img_list = glob.glob(os.path.join(config["data_save_dir"], "train/*/*"))

    df = pd.DataFrame(columns=["img_path", "label"])
    df["img_path"] = all_img_list
    df["label"] = df["img_path"].apply(lambda x: str(x).split("/")[-2])

    train_ds, val_ds, _, _ = train_test_split(
        df,
        df["label"],
        test_size=config["params"]["val_ratio"],
        stratify=df["label"],
        random_state=42,
    )

    le = preprocessing.LabelEncoder()
    train_ds["label"] = le.fit_transform(train_ds["label"])
    val_ds["label"] = le.transform(val_ds["label"])

    dataset_module = getattr(import_module("data"), config["dataset"])

    train_transform_module = getattr(import_module("data"), config["augment"]["train"])
    train_transform = train_transform_module(resize=config["augment"]["resize"])
    train_dataset = dataset_module(train_ds["img_path"].values, train_ds["label"].values, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["params"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    test_transform_module = getattr(import_module("data"), config["augment"]["test"])
    test_transform = test_transform_module(resize=config["augment"]["resize"])
    val_dataset = dataset_module(val_ds["img_path"].values, val_ds["label"].values, test_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["params"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    model_module = getattr(import_module("model"), config["model"])
    model = model_module()
    model.to("cuda")

    criterion = create_criterion(config["criterion"]["type"], **config["criterion"]["args"])
    opt_module = getattr(import_module("torch.optim"), config["optimizer"]["type"])
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        **config["optimizer"]["args"],
    )
    sch_module = getattr(import_module("torch.optim.lr_scheduler"), config["scheduler"]["type"])
    scheduler = sch_module(optimizer, **config["scheduler"]["args"])

    best_score = 0
    best_loss = np.inf
    patience = config["earlystop"]["patience"]
    counter = 0
    best_model = None

    model_path = make_model_path(config["model_save_dir"])
    with open(os.path.join(model_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    for epoch in range(1, config["params"]["epochs"] + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            model.eval()
            val_loss = []
            preds, true_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(iter(val_loader)):
                imgs = imgs.float().to("cuda")
                labels = labels.to("cuda")

                pred = model(imgs)

                loss = criterion(pred, labels)

                preds += pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += labels.detach().cpu().numpy().tolist()

                val_loss.append(loss.item())

            _val_loss = np.mean(val_loss)
            _val_score = f1_score(true_labels, preds, average="weighted")

        _train_loss = np.mean(train_loss)
        print(
            f"Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]"
        )

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_loss > _val_loss:
            print(f"New best model for val f1 : {_val_score:4.4%}, loss: {_val_loss:.5f}! saving the best model..")
            torch.save(model.state_dict(), f"{model_path}/best.pth")
            best_loss = _val_loss
            best_score = _val_score
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print(f"No validation performace improvement until {counter} iteration. Training stopped.")
            break

        # Wandb Logging for each epoch
        wandb.log({"Validation Score": _val_score, "Validation Loss": _val_loss, "Train Loss": _train_loss})

    print(f"Best loss and score is {best_loss}, and {best_score:4.4%}.")
    with open(os.path.join(model_path, "model_config.json"), "w") as f:
        model_conf = config
        model_conf["best_loss"] = best_loss
        model_conf["best_score"] = best_score
        json.dump(model_conf, f, indent=4)

    print(f"Best loss and score is {best_loss}, and {best_score:4.4%}.")


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
