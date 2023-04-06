from io import TextIOWrapper

class ConfigCaltech(object):
    def __init__(self):
        self.backbone  = 'ResNet50-CLIP-VLPD'
        # ResNet50-Concat (+self.score_map=False): Pytorch Implmentation of CSP
        # ResNet50-CLIP (+self.score_map=False): CLIP-initializaed CSP (CLIP+CSP)
        # ResNet50-CLIP (+self.score_map=True): CLIP+CSP+VLS
        # ResNet50-CLIP-VLPD (+sef.score_map=True): our proposed VLPD (CLIP+CSP+VLS+PSC)
        
        self.seed = 1337
        self.gen_seed = False
        
        # training config
        self.onegpu = 16
        self.num_epochs = 200
        self.add_epoch = 0
        self.iter_per_epoch = 2000 
        self.init_lr = 1e-4
        self.lr_policy = 'step'     #  or cyclic for SWA
        self.lr_step = [350]       #  no step
        self.warm_up = 3
        self.alpha = 0.999

        # dataset
        self.root_path = '/root/data/caltech/'     # the path to your caltech dataset  

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (336, 448)
        self.size_test = (480, 640)

        # image channel-wise mean to subtract, the order is BGR
        self.norm_mean = [123.675, 116.28, 103.53]
        self.norm_std = [58.395, 57.12, 57.375]

        self.log_freq = 10
        # whether or not to perform validation during training
        self.val = True
        self.val_frequency = 2
        self.val_begin = 70
        self.save_begin = 150
        self.save_end = 200

        self.score_map = True
        # whether ot not to use the strategy of weight moving average following CSP
        self.teacher = True     

        self.templates = ['a picture of {}']
        self.classnames = ['ground', 'building', 'tree', 'human', 'car', 'bus', 'bicycle', 'truck', 'traffic sign', 'sky']
        self.clip_weight = '/root/RN50.pt'
        
        self.seg_lambda = 1e2
        self.contrast_lambda = 1e-4

        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'h'  # or 'w', 'hw'
        self.num_scale = 1  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

    def print_conf(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))

    def write_conf(self, log: TextIOWrapper):
        log.write('\n'.join(['%s:%s' % item for item in self.__dict__.items()])+'\n')
        log.flush()