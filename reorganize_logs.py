import os
import shutil

def restructure_logs(src_dir, dst_dir):
    # Remove the existing clean_logs directory
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        if "arguments.txt" in files:
            with open(os.path.join(root, "arguments.txt"), "r") as f:
                args = dict(line.strip().split(": ") for line in f)

            env = args["env"]
            policy = args["policy"]
            run_folder = os.path.basename(root)

            new_path = os.path.join(dst_dir, os.path.basename(os.path.dirname(root)), env, policy, run_folder)
            os.makedirs(new_path, exist_ok=True)
            shutil.copytree(root, new_path, dirs_exist_ok=True)

if __name__ == "__main__":
    src_dir = "logs"
    dst_dir = "clean_logs"
    restructure_logs(src_dir, dst_dir)