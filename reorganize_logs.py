import os
import shutil
import argparse

def restructure_logs(src_dir, dst_dir, subfolder):
    # Remove the existing clean_logs directory
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    src_subfolder = os.path.join(src_dir, subfolder)

    for root, dirs, files in os.walk(src_subfolder):
        if "arguments.txt" in files:
            with open(os.path.join(root, "arguments.txt"), "r") as f:
                args = dict(line.strip().split(": ") for line in f)
            env = args["env"]
            policy = args["policy"]
            num_layers = args["num_conv_layers"]
            obs = args["obs"]
            run_folder = os.path.basename(root)
            new_path = os.path.join(dst_dir, subfolder, obs, env, policy + num_layers, run_folder)
            os.makedirs(new_path, exist_ok=True)
            shutil.copytree(root, new_path, dirs_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure logs")
    parser.add_argument("env", help="Subfolder in the 'logs' directory to read")
    args = parser.parse_args()

    src_dir = "logs"
    dst_dir = "clean_logs"
    subfolder = args.env

    restructure_logs(src_dir, dst_dir, subfolder)