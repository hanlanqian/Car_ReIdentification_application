import os
import re
import random
import argparse


def my_dataset2veri776(root_path, target_path):
    cid_map = {
        '1-1-971': 'c001',
        '1-2-981': 'c002',
        '5-1-1631': 'c003',
        '5-2-1641': 'c004',
    }
    fake_vid = 0
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    ## rename files
    for dir in os.listdir(root_path):
        for file in os.listdir(os.path.join(root_path, dir)):
            cid = cid_map[file.strip().split('_')[0]]
            os.rename(os.path.join(root_path, dir, file),
                      os.path.join(target_path, str(fake_vid) + '_' + cid + '_' + str(random.randint(0, 1e5)) + '.jpg'))
            fake_vid += 1

    ## make query, gallery
    _index = os.listdir(target_path)
    random.shuffle(_index)
    gallery_path = os.path.join(target_path, 'image_test')
    threshold = 1  # gallery rate
    query_path = os.path.join(target_path, 'image_query')
    if not os.path.exists(gallery_path):
        os.makedirs(gallery_path)
    if not os.path.exists(query_path):
        os.makedirs(query_path)
    for i, filename in enumerate(_index):
        if i / len(_index) <= threshold:
            os.rename(os.path.join(target_path, filename), os.path.join(gallery_path, filename))
        else:
            os.rename(os.path.join(target_path, filename), os.path.join(query_path, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../my_dataset_copy')
    parser.add_argument('--target', type=str, default='../../my_veri776')
    args = parser.parse_args()
    my_dataset2veri776(args.root, args.target)
