import os
import re


def clean_epoch_files(project_root, relative_path):
    epoch_pattern = re.compile(r'^epoch_(\d+)\.pth$')
    directory = os.path.join(project_root, relative_path)

    max_epoch = -1
    max_file_path = None

    for filename in os.listdir(directory):
        match = epoch_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_file_path = os.path.join(directory, filename)

    if max_file_path is not None:
        print(f"Keeping the file with the latest epoch: {max_file_path}")
        for filename in os.listdir(directory):
            if epoch_pattern.match(filename) and filename != os.path.basename(max_file_path):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    else:
        print("No max epoch model weights match the pattern.")


if __name__ == '__main__':
    word_dir = "../work_dirs/tiny/clip/simple-cross-attn-category/split3"
    clean_epoch_files(word_dir)