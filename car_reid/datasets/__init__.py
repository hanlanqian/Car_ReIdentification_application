from . import datasets
from . import demo_transforms as demo_trans


def make_basic_dataset(metas, val_size, pad, *, test_ext='', re_prob=0.5, with_mask=False, for_vis=False,
                       infer_flag=False):
    """
    构建基础数据集。
    pkl_path -> metas
    """

    meta_dataset = datasets.CommonReIDDataset(metas=metas, test_ext=test_ext, infer=infer_flag)
    val_transform = demo_trans.get_validation_augmentations(val_size)
    if for_vis:
        preprocessing = None
    else:
        # baiyan model
        preprocessing = demo_trans.get_preprocessing(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    val_dataset = datasets.ReIDDataset(meta_dataset.query + meta_dataset.gallery, with_mask=with_mask,
                                       transform=val_transform,
                                       preprocessing=preprocessing)

    return val_dataset, meta_dataset
