# Third Party
import argparse
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch

# In House
from aeromatch.config.settings import read_settings_file
from aeromatch.data.load_tartandrive2_dataset import TartanDrive
from aeromatch.models.bev_loc import BEVLoc

# Entrypoint:
if __name__ == "__main__":
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Create argument parser
    parser = argparse.ArgumentParser("TartanDrive Trainer", None, "TartanDriver Trainer for BEVLoc")
    parser.add_argument("-s", "--settings", default="aeromatch/config/tartandrive_settings_lowmem.json")
    args = parser.parse_args()

    # Get settings for dataset
    tartan_drive_settings = read_settings_file(args.settings)
    
    #* Create BEVLoc and pass in all the relevant modules
    bev_loc = BEVLoc(tartan_drive_settings)
    # BEVLoc.load_from_checkpoint("/home/chris/dev/tartan-aeromatch/models/ResNetFine.ckpt", tartan_drive_settings=tartan_drive_settings)

    #* Parse a file to what is wanted for our PyTorch dataset
    train_dir = tartan_drive_settings["training"]["train_split"]
    val_dir   = tartan_drive_settings["training"]["val_split"]
    test_dir  = tartan_drive_settings["training"]["test_split"]
    train_traj_dirs     = [f"{train_dir}/{fp}" for fp in os.listdir(train_dir) if os.path.isdir(f"{train_dir}/{fp}")]
    train_drive_dataset = TartanDrive(train_traj_dirs, bev_loc.get_aero_bev(), tartan_drive_settings)
    val_traj_dirs       = [f"{val_dir}/{fp}" for fp in os.listdir(val_dir) if os.path.isdir(f"{val_dir}/{fp}")]
    val_drive_dataset   = TartanDrive(val_traj_dirs, bev_loc.get_aero_bev(), tartan_drive_settings)
    print(f"Number of Training Trajectories in Dataset: {len(train_traj_dirs)}")

    # Path specifics
    checkpoint_out = tartan_drive_settings["checkpoint_out_path"]

    # Training loop
    train_loader = DataLoader(train_drive_dataset, batch_size=tartan_drive_settings["training"]["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_drive_dataset, batch_size=tartan_drive_settings["training"]["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    # Define the ModelCheckpoint callback
    # import os
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='train_loss',  # Specify the metric to monitor for saving checkpoints
    #     dirpath=os.getcwd(),
    #     filename="checkpoint-{checkpoint_root}-{epoch:02d}",
    #     save_top_k=-1,  # Save the top 3 models with the best validation loss
    # )

    # Create a PyTorch Lightning Trainer with the ModelCheckpoint callback
    logger = WandbLogger("BEVLoc")
    trainer = pl.Trainer(
        max_epochs=10,  # Specify the number of epochs
        default_root_dir=os.getcwd(),
        check_val_every_n_epoch=10,
        logger=logger
    )

    # Fit the model
    trainer.fit(bev_loc, train_dataloaders=train_loader)
    trainer.save_checkpoint("training.ckpt")
    torch.cuda.empty_cache()
