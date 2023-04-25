from importlib import import_module
import pandas as pd
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score
import json
from loss import create_criterion
from datetime import datetime
import os
import wandb

def make_model_path(base_path):
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d-%H:%M:%S")
    model_path = os.path.join(base_path, dt_string)
    os.makedirs(model_path, exist_ok=False)
    return model_path

def main(config):
    #Wandb Initialization
    wandb.init(
        project = 'papering',
        
        config = {
            "Model" : config['model'],
            "Dataset" : config['dataset'],
            "Optimizer" : config['optimizer']['type'],
            "Learning Rate" : config['optimizer']['args']['lr'],
            "Scheduler" : config['scheduler']['type'],
            "Epoch" : config['params']['epochs'],
            "Batch Size" : config['params']['batch_size']
        }
    )
    
    model_path = make_model_path(config["model_save_dir"])
    all_img_list = glob.glob('./dataset/train/*/*')

    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])

    train_ds, val_ds, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=42)

    le = preprocessing.LabelEncoder()
    train_ds['label'] = le.fit_transform(train_ds['label'])
    val_ds['label'] = le.transform(val_ds['label'])
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])

    dataset_module = getattr(import_module("data"), config["dataset"])
    
    train_dataset = dataset_module(train_ds['img_path'].values, train_ds['label'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["params"]["batch_size"], shuffle=False, num_workers=4)

    val_dataset = dataset_module(val_ds['img_path'].values, val_ds['label'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=config["params"]["batch_size"], shuffle=False, num_workers=4)
    
    model_module = getattr(import_module("model"), config["model"])
    model = model_module()
    model.to("cuda")
    
    criterion = create_criterion(config["criterion"]["type"], **config["criterion"]["args"])
    opt_module = getattr(import_module("torch.optim"), config["optimizer"]["type"])
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        **config["optimizer"]["args"]
    )
    sch_module = getattr(import_module("torch.optim.lr_scheduler"), config["scheduler"]["type"])
    scheduler = sch_module(
        optimizer, **config["scheduler"]["args"]
    )
    
    best_score = 0
    best_loss = np.inf
    patience = config["earlystop"]["patience"]
    counter = 0
    best_model = None
    
    for epoch in range(1, config["params"]["epochs"]+1):
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
            _val_score = f1_score(true_labels, preds, average='weighted')
                    
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
       
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
        
        if counter == config["earlystop"]["patience"]:
            print(f"No validation performace improvement until {counter} iteration. Training stopped.")
            break
            
        #Wandb Logging for each epoch
        wandb.log({
            'Validation Score':_val_score,
            'Validation Loss': _val_loss
        })
        
    print("Best loss and score is {best_loss}, and {best_score:4.4%}.")


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    main(config)