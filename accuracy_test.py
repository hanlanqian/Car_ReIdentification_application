import torch
from car_reid.configs.default_config import cfg
from car_reid.generate_pkl import veri776
from car_reid.inference import inference
from car_reid.reid_utils.iotools import merge_configs

cfg = merge_configs(cfg, 'car_reid/configs/eval.yml')
metas = veri776(r"C:\Users\hanlanqian\Desktop\车辆重识别\test_processed", cfg.eval.pkl_file)

parsing_model = torch.load(cfg.parse.model_path)
parsing_model = parsing_model.cuda()
parsing_model.eval()
output_path = Path(cfg.eval.output).absolute()
for phase in metas.keys():
    sub_path = output_path / phase
    mkdir_p(str(sub_path))
    dataset = VehicleReIDParsingDataset(metas[phase], augmentation=get_validation_augmentation(),
                                        preprocessing=get_preprocessing(
                                            smp.encoders.get_preprocessing_fn(cfg.encoder, cfg.encoder_weights)))
    dataset_vis = VehicleReIDParsingDataset(metas[phase], with_extra=True)
    signals[1].emit(f'Predict {phase} mask to {sub_path}')
    mask_predict(parsing_model, dataset, dataset_vis, sub_path, phase, signals[0])
