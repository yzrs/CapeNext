from xtcocotools import coco
import os
import shutil


def check_img(split):
    base_dir = 'dense-mp-100/mp-100'
    train_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_train.json')
    train_coco = coco.COCO(train_label_path)
    val_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_val.json')
    val_coco = coco.COCO(val_label_path)
    test_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_test.json')
    test_coco = coco.COCO(test_label_path)
    cocos = [train_coco, val_coco, test_coco]
    pre_dir = set()
    for cocoDataset in cocos:
        imgs = cocoDataset.loadImgs(cocoDataset.getImgIds())
        for img in imgs:
            file_name = img['file_name']
            split_res = file_name.split('/')
            if len(split_res) != 2:
                # print("file_name: {}".format(file_name))
                pre_dir.add(split_res[0] + '/' + split_res[1] + '/' + split_res[2])
            else:
                pre_dir.add(split_res[0])
            img_path = os.path.join(base_dir, file_name)
            os.path.relpath(img_path)
            if not os.path.exists(img_path):
                print("{} not exist!".format(img_path))
    return pre_dir


def dense_mp100(split, base_dir='../dataset/mp-100', dst_dir='./dense-mp-100'):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    train_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_train.json')
    train_coco = coco.COCO(train_label_path)
    val_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_val.json')
    val_coco = coco.COCO(val_label_path)
    test_label_path = os.path.join(base_dir, 'annotations', 'mp100_split' + split + '_test.json')
    test_coco = coco.COCO(test_label_path)
    cocos = [train_coco, val_coco, test_coco]
    pre_dir = set()
    prefix = '/home/zy/sust/dataset/'
    for cocoDataset in cocos:
        imgs = cocoDataset.loadImgs(cocoDataset.getImgIds())
        for img in imgs:
            file_name = img['file_name']
            split_res = file_name.split('/')
            if len(split_res) != 2:
                # print("file_name: {}".format(file_name))
                pre_dir.add(split_res[0] + '/' + split_res[1] + '/' + split_res[2])
            else:
                pre_dir.add(split_res[0])
            img_path = os.path.join(base_dir, file_name)
            actual_img_path = os.path.realpath(img_path)
            if not os.path.exists(img_path):
                print("{} not exist!".format(img_path))
            else:
                # actual_img_path - prefix
                rel_path = actual_img_path.replace(prefix, '')
                dst_img_path = os.path.join(dst_dir, rel_path)
                if not os.path.exists(dst_img_path):
                    dst_img_dir = os.path.dirname(dst_img_path)
                    os.makedirs(dst_img_dir, exist_ok=True)
                    shutil.copy2(actual_img_path, dst_img_path)
                    print(f"Split{split}: Copied {actual_img_path} to {dst_img_path}")
                else:
                    print(f"Split{split}: {actual_img_path} already exists at {dst_img_path}")

    return pre_dir


def symlink(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件和目录（包括软链接）
    for root, dirs, files in os.walk(source_folder):
        relative_root = os.path.relpath(root, source_folder)
        target_root = os.path.join(destination_folder, relative_root)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        for file in dirs:
            file_path = os.path.join(root, file)
            # 判断是否为软链接
            if os.path.islink(file_path):
                link_destination = os.readlink(file_path)
                target_file_path = os.path.join(target_root, file)
                # 创建软链接到目标文件夹中对应的位置
                os.symlink(link_destination, target_file_path)


if __name__ == '__main__':
    # symlink(source_folder='../dataset/mp-100', destination_folder='./dense-mp-100/mp-100')
    splits = ['1', '2', '3', '4', '5']
    # splits = ['1']
    for split in splits:
    #     pre_dir = dense_mp100(split)
        pre_dir = check_img(split)
    pre_dir = sorted(list(pre_dir))
    print(pre_dir)
    # with open('pre_dir.txt', 'w') as f:
    #     for pre in pre_dir:
    #         f.write(pre + '\n')