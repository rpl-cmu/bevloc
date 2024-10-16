# Third Party
import argparse
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb

# In House
from aeromatch.config.settings import read_settings_file
from aeromatch.data.load_tartandrive2_dataset import TartanDrive
from aeromatch.models.bev_loc import BEVLoc

# Entrypoint:
if __name__ == "__main__":
    # Create argument parser
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser("TartanDrive Trainer", None, "TartanDriver Trainer for BEVLoc")
    parser.add_argument("-s", "--settings", default="aeromatch/config/predator.json")
    args = parser.parse_args()

    # Get settings for dataset
    tartan_drive_settings = read_settings_file(args.settings)
    
    #* Create BEVLoc and pass in all the relevant modules
    path = "models"
    model_list = [f"{path}/DINOG.ckpt"]
    for mc in model_list:
        model = BEVLoc.load_from_checkpoint(mc, tartan_drive_settings=tartan_drive_settings)
        model.eval()

        #* Parse a file to what is wanted for our PyTorch dataset
        test_dir  = tartan_drive_settings["training"]["val_split"]
        test_traj_dirs       = [f"{test_dir}/{fp}" for fp in os.listdir(test_dir) if os.path.isdir(f"{test_dir}/{fp}")]
        print(test_traj_dirs)
        for test_traj_dir in test_traj_dirs:
            test_drive_dataset   = TartanDrive([test_traj_dir], model.get_aero_bev(), tartan_drive_settings)
            print(f"Number of Testing Trajectories: {len(test_traj_dirs)}")

            # Path specifics
            checkpoint_out = tartan_drive_settings["checkpoint_out_path"]
            
            # Model string
            mc_str = mc.split("/")[-1].split(".")[0]
            traj_stem = test_traj_dir.split("/")[-1]
            print(f"{mc_str}_{traj_stem}")
            model.model = f"{mc_str}_{traj_stem}"

            # Test/Evaluate on TartanDrive
            traj_loader = DataLoader(test_drive_dataset, batch_size=8, shuffle=False, num_workers=1)
            logger = WandbLogger("BEVLoc", log_model=mc)
            trainer = pl.Trainer(logger=logger)
            trainer.test(model=model, dataloaders=traj_loader)
            wandb.finish()
            torch.cuda.empty_cache()
