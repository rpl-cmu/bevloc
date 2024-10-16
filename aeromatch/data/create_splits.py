# Third Party
import os
import argparse
import json
import random

# In House


def create_splits(trajs, train_pct, val_pct, test_pct, train, val, test, exclude):
    """
    The idea here is to have a json file that had set in stone trajectories we want in the train, val,and test sets.
    Anything else will be randomly put in a set based on the desired percentages.
    """
    remaining_traj = set(trajs) - set(train) - set(val) - set(test) - set(exclude)
    num_train_add = round(len(remaining_traj) * train_pct - len(train))
    num_val_add   = round(len(remaining_traj) * val_pct - len(val))
    num_test_add  = round(len(remaining_traj) * test_pct - len(test))

    # Sample and replace
    train_elements = random.sample(remaining_traj, k = num_train_add)
    remaining_traj = remaining_traj - set(train_elements)
    test_elements  = random.sample(remaining_traj, k = num_test_add)
    remaining_traj = remaining_traj - set(test_elements)
    
    train += train_elements
    val += remaining_traj
    test += test_elements
    return train, val, test

def create_symlinks(inp, out, train, val, test):
    for t in train:
        os.symlink(f"{inp}/{t}", f"{out}/train/{t}", target_is_directory=True)
    for v in val:
        os.symlink(f"{inp}/{v}", f"{out}/val/{v}", target_is_directory=True)
    for te in test:
        os.symlink(f"{inp}/{te}", f"{out}/test/{te}", target_is_directory=True)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser("TartanDrive Split Creator", None, "TartanDriver Split Creator for BEVLoc")
    parser.add_argument("-i", "--input_folder", default="/media/klammerc/starscream_data/tartandrive/2022_traj")
    parser.add_argument("-o", "--output_folder", default="/media/klammerc/starscream_data/tartandrive/2022_split")
    parser.add_argument("-j", "--json_file", default="aeromatch/data/splits.json")
    args = parser.parse_args()

    # Create split dirs
    train_dir = f"{args.output_folder}/train"
    val_dir   = f"{args.output_folder}/val"
    test_dir  = f"{args.output_folder}/test"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Read JSON data
    json_data = None
    with open(args.json_file, "r") as f:
        json_data = json.load(f)

    # Get all of the possible trajs
    trajs = os.listdir(args.input_folder)

    # Create the splits
    train = json_data["train"]
    val   = json_data["val"]
    test  = json_data["test"]
    exclude = json_data["exclude"]
    train_split, val_split, test_split = create_splits(trajs, json_data["train_pct"], json_data["val_pct"], json_data["test_pct"], train, val, test, exclude)

    # Create the symlinks
    create_symlinks(args.input_folder, args.output_folder, train_split, val_split, test_split)