import argparse
import glob
import json
import os
from importlib import import_module

import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import seed_everything


def inference(model, le, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            pred = model(imgs)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()

    preds = le.inverse_transform(preds)
    return preds


def main(args):
    model_dir = args.model_dir
    with open(os.path.join(model_dir, "model_config.json")) as f:
        config = json.load(f)

    def expand_path(path_dir):
        data_dir = config["data_save_dir"]
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]
        path_dir = path_dir.replace("./", data_dir + "/")
        return path_dir

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
    le.fit_transform(train_ds["label"])

    test = pd.read_csv(os.path.join(config["data_save_dir"], "test.csv"))

    test["img_path"] = test["img_path"].apply(expand_path)

    dataset_module = getattr(import_module("data"), config["dataset"])
    test_transform_module = getattr(import_module("data"), config["augment"]["test"])
    test_transform = test_transform_module(resize=config["augment"]["resize"])
    print(test["img_path"])
    test_dataset = dataset_module(test["img_path"].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config["params"]["batch_size"], shuffle=False, num_workers=4)

    checkpoint = torch.load(os.path.join(model_dir, "best.pth"))
    model_module = getattr(import_module("model"), config["model"])
    infer_model = model_module()
    infer_model.load_state_dict(checkpoint)
    infer_model.to("cuda")

    preds = inference(infer_model, le, test_loader, "cuda")

    submit = pd.read_csv(os.path.join(config["data_save_dir"], "sample_submission.csv"))

    submit["label"] = preds

    submit.to_csv("./baseline_submit.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help='model directory to load model and conf json file. for example, "--model_dir model/2023.01.01-12:01:23" or something like this.',
    )

    args = parser.parse_args()

    main(args)
