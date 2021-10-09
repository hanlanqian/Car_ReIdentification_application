import pandas as pd
import os
import pickle as pkl
from pathlib import Path
import re


def veri776(input_path, output_path, inference):
    input_path = os.path.abspath(input_path)
    output_dir, output_filename = os.path.split(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_path).absolute()
    output_dict = {}

    pattern = re.compile(r"(\d+)_(\d+)_c(\d+).*\.jpg")
    if not inference:
        dir_set = ["train", "query", "gallery"]
    else:
        dir_set = ['query', 'gallery']
    for phase in dir_set:
        output_dict[phase] = []
        sub_path = input_path / f"image_{phase}"
        if phase == "gallery":
            sub_path = input_path / f"image_test"
        for image_path in sub_path.iterdir():
            sample = {}
            image_name = image_path.name
            result = pattern.match(image_name)
            if not result:
                continue
            id, obj_id, camera = result.groups()
            sample["filename"] = image_name
            sample["image_path"] = str(image_path)
            sample["id"] = id
            sample["object_id"] = obj_id
            sample["cam"] = camera
            output_dict[phase].append(sample)
    print(output_path)
    with open(output_path, "wb") as f:
        pkl.dump(output_dict, f)
    return output_dict


def vehicleid(input_path, output_path):
    input_path = os.path.abspath(input_path)
    PATH = input_path

    images = {}

    images['train'] = open(PATH + '/train_test_split/train_list.txt').read().strip().split('\n')
    images['gallery_800'] = open(PATH + '/train_test_split/test_list_800.txt').read().strip().split('\n')
    images['gallery_1600'] = open(PATH + '/train_test_split/test_list_1600.txt').read().strip().split('\n')
    images['gallery_2400'] = open(PATH + '/train_test_split/test_list_2400.txt').read().strip().split('\n')
    images['query_800'] = []
    images['query_1600'] = []
    images['query_2400'] = []

    outputs = {}
    for key, lists in images.items():
        output = []
        for img_name in lists:
            item = {
                "image_path": f"{PATH}/image/{img_name.split(' ')[0]}.jpg",
                "name": img_name,
                "id": img_name.split(' ')[1],
                "cam": 0
            }
            output.append(item)
        outputs[key] = output

    base_path = os.path.split(output_path)[0]
    if base_path != '' and not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    with open(output_path, 'wb') as f:
        pkl.dump(outputs, f)


def veriwild(input_path, output_path):
    input_path = os.path.abspath(input_path)
    PATH = input_path

    images = {}

    images['train'] = open(PATH + '/train_test_split/train_list.txt').read().strip().split('\n')
    images['query_3000'] = open(PATH + '/train_test_split/test_3000_query.txt').read().strip().split('\n')
    images['gallery_3000'] = open(PATH + '/train_test_split/test_3000.txt').read().strip().split('\n')
    images['query_5000'] = open(PATH + '/train_test_split/test_5000_query.txt').read().strip().split('\n')
    images['gallery_5000'] = open(PATH + '/train_test_split/test_5000.txt').read().strip().split('\n')
    images['query_10000'] = open(PATH + '/train_test_split/test_10000_query.txt').read().strip().split('\n')
    images['gallery_10000'] = open(PATH + '/train_test_split/test_10000.txt').read().strip().split('\n')

    wild_df = pd.read_csv(f'{PATH}/train_test_split/vehicle_info.txt', sep=';', index_col='id/image')

    # Pandas indexing is very slow, change it to dict
    wild_dict = wild_df.to_dict()
    camid_dict = wild_dict['Camera ID']

    outputs = {}
    for key, lists in images.items():
        output = []
        for img_name in lists:
            item = {
                "image_path": f"{PATH}/images/{img_name}.jpg",
                "name": img_name,
                "id": img_name.split('/')[0],
                #             "cam": wild_df.loc[img_name]['Camera ID']
                "cam": camid_dict[img_name]
            }
            output.append(item)
        outputs[key] = output

    base_path = os.path.split(output_path)[0]
    if base_path != '' and not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    with open(output_path, 'wb') as f:
        pkl.dump(outputs, f)
