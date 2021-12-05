from yacs.config import CfgNode

cfg = CfgNode()
cfg.desc = ""  # 对本次实验的简单描述，用于为tensorboard命名
cfg.stage = "train"  # train or eval or test
cfg.device = "cpu"  # cpu or cuda
cfg.device_ids = ""  # if not set, use all gpus
cfg.output_dir = "/data/vehicle_reid/perspective_transform_feature/debug"
cfg.debug = False

cfg.train = CfgNode()
cfg.train.epochs = 120

cfg.data = CfgNode()
cfg.data.name = "VeRi776"
cfg.data.pkl_path = "../data_processing/veri776.pkl"
cfg.data.train_size = (256, 256)
cfg.data.valid_size = (256, 256)
cfg.data.pad = 10
cfg.data.re_prob = 0.5
cfg.data.with_mask = True
cfg.data.test_ext = ''

cfg.data.sampler = 'RandomIdentitySampler'
cfg.data.batch_size = 16
cfg.data.num_instances = 4

cfg.data.train_num_workers = 0
cfg.data.test_num_workers = 0

cfg.model = CfgNode()
cfg.model.name = "resnet50"
# If it is set to empty, we will download it from torchvision official website.
cfg.model.pretrain_path = ""
cfg.model.last_stride = 1
cfg.model.neck = 'bnneck'
cfg.model.neck_feat = 'after'
cfg.model.pretrain_choice = 'imagenet'
cfg.model.ckpt_period = 10

cfg.optim = CfgNode()
cfg.optim.name = 'Adam'
cfg.optim.base_lr = 3.5e-4
cfg.optim.bias_lr_factor = 1
cfg.optim.weight_decay = 0.0005
cfg.optim.momentum = 0.9

cfg.loss = CfgNode()
cfg.loss.losses = ["triplet", "id", "center", "local-triplet"]
cfg.loss.triplet_margin = 0.3
cfg.loss.normalize_feature = True
cfg.loss.id_epsilon = 0.1

cfg.loss.center_lr = 0.5
cfg.loss.center_weight = 0.0005

cfg.loss.tuplet_s = 64
cfg.loss.tuplet_beta = 0.1

cfg.scheduler = CfgNode()
cfg.scheduler.milestones = [40, 70]
cfg.scheduler.gamma = 0.1
cfg.scheduler.warmup_factor = 0.0
cfg.scheduler.warmup_iters = 10
cfg.scheduler.warmup_method = "linear"

cfg.test = CfgNode()
cfg.test.feat_norm = True
cfg.test.remove_junk = True
cfg.test.period = 10
cfg.test.device = "cuda"
cfg.test.model_path = "./car_reid/checkpoints/veri776_reid.pth"
cfg.test.color_clf_model_path = './car_reid/checkpoints/color_model.pth'
cfg.test.vehicle_clf_model_path = './car_reid/checkpoints/vehicle_type.pth'
cfg.test.color_type_json = './car_reid/configs/color.json'
cfg.test.vehicle_type_json = './car_reid/configs/vehicle.json'
cfg.test.max_rank = 50
cfg.test.rerank = False
cfg.test.lambda_ = 0.0
cfg.test.infer_flag = False  ## train flag
# split: When the CUDA memory is not sufficient,
# we can split the dataset into different parts
# for the computing of distance.
cfg.test.split = 0

cfg.parse = CfgNode()
cfg.parse.model_path = './car_reid/checkpoints/parsing_model.pth'


cfg.logging = CfgNode()
cfg.logging.level = "info"
cfg.logging.period = 20

## parsing config
cfg.encoder = 'se_resnext50_32x4d'
cfg.encoder_weights = 'imagenet'

## eval part
cfg.eval = CfgNode()
cfg.eval.pkl_file = ''
cfg.eval.output = './outputs/'

