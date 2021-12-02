from datasets.datasets import VehicleReIDParsingDataset, get_preprocessing, get_validation_augmentation
from pathlib import Path
from reid_utils import mkdir_p
from yacs.config import CfgNode
from logzero import logger
from reid_models import ParsingReidModel, resnet34, resnet50
from reid_utils.math_tools import Clck_R1_mAP
from reid_utils.iotools import merge_configs
from generate_pkl import veri776
from configs.default_config import cfg
from datasets import make_basic_dataset
from torch.utils.data import DataLoader

import torch
import logging
import argparse
import pickle
import cv2
import os
import numpy as np
import json
import configparser
import segmentation_models_pytorch as smp

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

IMG2MASK = {}


def mask_predict(model, test_dataset, test_dataset_vis, output_path, flag, processbar_signal):
    mkdir_p(output_path)
    if flag == 'query':
        start = 0
        percent = 10
    elif flag == 'gallery':
        start = 10
        percent = 30
    for i in range(len(test_dataset)):
        processbar_signal.emit(int(start + i / len(test_dataset) * percent))
        image = test_dataset[i]
        image_vis, extra = test_dataset_vis[i]

        ## 重复图片直接用之前计算好的即可
        image_path = Path(extra["image_path"])
        # if str(image_path) in IMG2MASK:
        #     extra["mask_path"] = str(IMG2MASK[str(image_path)])
        #     continue
        mask_path = output_path / f"{image_path.name.split('.')[0]}.png"
        #
        # x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        # with torch.no_grad():
        #     pr_mask = model.predict(x_tensor)
        # pr_map = pr_mask.squeeze().cpu().numpy().round()
        # pr_map = np.argmax(pr_map, axis=0)[:image_vis.shape[0], :image_vis.shape[1]]
        # cv2.imwrite(str(mask_path), pr_map.astype(np.uint8))
        extra["mask_path"] = str(mask_path)

        IMG2MASK[str(image_path)] = str(mask_path)


def build_model(cfg, num_classes):
    ## 构建并返回返回parse_reid, color 和 type网络模型
    # 读取预测种类
    def get_num_classes(json_path):
        with open(json_path, 'r') as f:
            classes = json.load(f)
        return classes

    reid_model = ParsingReidModel(num_classes, cfg.model.last_stride, cfg.model.pretrain_path, cfg.model.neck,
                                  cfg.model.neck_feat, cfg.model.name, cfg.model.pretrain_choice).to(cfg.device)

    color_model = resnet34(num_classes=len(get_num_classes(cfg.test.color_type_json))).to(cfg.device)
    type_model = resnet50(num_classes=len(get_num_classes(cfg.test.vehicle_type_json))).to(cfg.device)
    return reid_model, color_model, type_model


def models_weight_load(cfg, models):
    ## parsing reid network weight loading
    state_dict = torch.load(cfg.test.model_path, map_location=cfg.device)
    remove_keys = []

    for key, value in state_dict.items():
        if 'classifier' in key:
            remove_keys.append(key)
    for key in remove_keys:
        del state_dict[key]
    models[0].load_state_dict(state_dict, strict=False)
    if torch.cuda.device_count() > 1:
        models[0] = torch.nn.DataParallel(models[0])

    ## color classifier network weight loading
    assert os.path.exists(cfg.test.color_clf_model_path), "file: '{}' dose not exist.".format(
        cfg.test.color_clf_model_path)
    models[1].load_state_dict(torch.load(cfg.test.color_clf_model_path, map_location=cfg.device))

    assert os.path.exists(cfg.test.vehicle_clf_model_path), "file: '{}' dose not exist.".format(
        cfg.test.vehicle_clf_model_path)
    models[2].load_state_dict(torch.load(cfg.test.vehicle_clf_model_path, map_location=cfg.device))


def eval_(models,
          device,
          valid_loader,
          query_length,
          feat_norm=True,
          remove_junk=True,
          max_rank=50,
          output_dir='',
          rerank=False,
          conf=None,
          split=0,
          output_html_path='',
          **kwargs):
    metric = Clck_R1_mAP(query_length, max_rank=max_rank, rerank=rerank, remove_junk=remove_junk, feat_norm=feat_norm,
                         output_path=output_dir, alpha=conf.getfloat('reid', 'ALPHA'),
                         beta=conf.getfloat('reid', 'BETA'),
                         lambda1=conf.getfloat('reid', 'LAMBDA1'), lambda2=conf.getfloat('reid', 'LAMBDA2'))
    [model.eval() for model in models]
    processbar_singal = kwargs.get('signals', None)[0]
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            processbar_singal.emit(int(40 + (i / len(valid_loader)) * 60))
            for name, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[name] = item.to("cuda")
            output = models[0](**batch)
            color_output = models[1](batch['image'])
            vehicle_type_output = models[2](batch['image'])
            global_feat = output["global_feat"]
            local_feat = output["local_feat"]
            vis_score = output["vis_score"]
            metric.update((global_feat.detach().cpu(), local_feat.detach().cpu(), vis_score.cpu(), batch["id"].cpu(),
                           batch["cam"].cpu(), batch["image_path"], color_output, vehicle_type_output))

    info_signal = kwargs.get('signals', None)[1]
    metric.save(f'{output_dir}/test_features.pkl')

    metric_output = metric.compute(split=split, infer_flag=conf.getboolean('default', 'infer_flag'))
    info_signal.emit(f'重识别结果已写入{output_dir}/test_output.pkl')
    processbar_singal.emit(100)
    if not kwargs['infer_flag']:
        cmc = metric_output['cmc']
        mAP = metric_output['mAP']
        final_distmat = metric_output['final_distmat']
        all_AP = metric_output['all_AP']
        metric.reset()
        info_signal.emit(f"mAP: {mAP:.2%}")
        for r in [1, 5, 10]:
            info_signal.emit(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.2%}")
        return cmc, mAP
    else:
        return metric_output.get('distmat')


def inference(conf: configparser.ConfigParser, cfg, signals=None):
    cfg = merge_configs(cfg, conf.get('reid', 'CONFIG_FILE'))
    os.makedirs(conf.get('reid', 'OUTPUT'), exist_ok=True)
    signals[1].emit("开始生成数据集信息pkl文件")
    metas = veri776(conf.get('video_process', 'OUTPUT'), conf.get('reid', 'PKL_PATH'), inference=True)

    signals[1].emit('开始生成图像masks')
    #### Stage one: generate masks ####
    parsing_model = torch.load(cfg.parse.model_path)
    parsing_model = parsing_model.cuda()
    parsing_model.eval()

    output_path = Path(conf.get('reid', 'OUTPUT')).absolute()

    for phase in metas.keys():
        sub_path = output_path / phase
        mkdir_p(str(sub_path))
        dataset = VehicleReIDParsingDataset(metas[phase], augmentation=get_validation_augmentation(),
                                            preprocessing=get_preprocessing(
                                                smp.encoders.get_preprocessing_fn(cfg.encoder, cfg.encoder_weights)))
        dataset_vis = VehicleReIDParsingDataset(metas[phase], with_extra=True)
        signals[1].emit(f'Predict {phase} mask to {sub_path}')
        mask_predict(parsing_model, dataset, dataset_vis, sub_path, phase, signals[0])
    signals[1].emit('车辆mask已全部生成')

    #### Stage Two: reid base on masks, color and type
    ## 清空cuda缓存
    torch.cuda.empty_cache()
    signals[1].emit(f"加载重识别模型 {cfg.test.model_path}")
    models = build_model(cfg, 1)
    models_weight_load(cfg, models)

    ## 创建数据集模型
    valid_dataset, meta_dataset = make_basic_dataset(metas,
                                                     cfg.data.train_size,
                                                     cfg.data.valid_size,
                                                     cfg.data.pad,
                                                     test_ext=cfg.data.test_ext,
                                                     re_prob=cfg.data.re_prob,
                                                     with_mask=cfg.data.with_mask,
                                                     infer_flag=conf.getboolean('default', 'infer_flag')
                                                     )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.data.batch_size,
                              num_workers=cfg.data.test_num_workers,
                              pin_memory=True,
                              shuffle=False)

    query_length = meta_dataset.num_query_imgs
    signals[1].emit("开始进行重识别推理.....")
    outputs = eval_(models, cfg.test.device, valid_loader, query_length,
                    feat_norm=cfg.test.feat_norm,
                    remove_junk=cfg.test.remove_junk,
                    max_rank=cfg.test.max_rank,
                    output_dir=conf.get('reid', 'OUTPUT'),
                    conf=conf,
                    rerank=cfg.test.rerank,
                    split=cfg.test.split,
                    output_html_path=cfg.test.output_html_path,
                    signals=signals)
