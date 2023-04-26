
import wandb



def main():
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
    )
