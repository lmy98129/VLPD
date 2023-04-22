import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.detector import CSP
from config import Config
from config_caltech import ConfigCaltech
from dataloader.loader import *
from utils.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
import numpy as np

import json
import argparse
import pdb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-path', default='', type=str, metavar='VAL_PATH', 
        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out-path', default='', type=str, metavar='OUT_PATH',
        help='path to save detection results in json format (city) or txt file folder (caltech)')
    parser.add_argument('--dataset', default='city', type=str, metavar='DATASET',
        help='dataset to choose, including CityPersons (city) and Caltech (caltech)')
    args = parser.parse_args()
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse()
    cfg = Config() if args.dataset == 'city' else ConfigCaltech()

    fix_seed(cfg.seed)

    print('Net Initializing')
    net = CSP(cfg).cuda()
    net = nn.DataParallel(net)

    checkpoint = torch.load(args.val_path)
    net.module.load_state_dict(checkpoint['model'])

    # dataset
    print('Dataset...')

    if args.dataset == 'city':
        testdataset = CityPersons(path=cfg.root_path, type='val', config=cfg)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=4)
        val_city(testloader, net, cfg, args)
    else:
        testdataset = Caltech(path=cfg.root_path, type='test', config=cfg)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=4)
        val_caltech(testloader, net, cfg, args)

def val_city(testloader, net, config, args):
    net.eval()

    print('Perform validation...')
    temp_val = args.out_path
    is_inference = not os.path.exists(temp_val)

    if is_inference:
        res = []
        inference_time = 0
        num_images = len(testloader)
        for i, data in enumerate(testloader):
            inputs = data.cuda()
            with torch.no_grad():
                t1 = time.time()
                pos, height, offset = net(inputs)
                t2 = time.time()
                inference_time += (t2 - t1)

            boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), 
                                config.size_test, score=0.1, down=4, nms_thresh=0.5)
            if len(boxes) > 0:
                boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                for box in boxes:
                    temp = dict()
                    temp['image_id'] = i+1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)

            print('\r%d/%d' % (i + 1, num_images),end='')
            sys.stdout.flush()

        with open(temp_val, 'w') as f:
            json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', temp_val)
    print('\nSummerize:[Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    if is_inference:
        FPS = int(num_images / inference_time)
        print('FPS : {}'.format(FPS))

def val_caltech(testloader: DataLoader, net, config: ConfigCaltech, args):
    net.eval()

    print('Perform validation...')
    t3 = time.time()
    
    res_path = args.out_path
    if not os.path.exists(res_path): os.mkdir(res_path)
    
    for st in range(6, 11):
        set_path = os.path.join(res_path, 'set' + '%02d' % st)
        if not os.path.exists(set_path): os.mkdir(set_path)

    testdataset: Caltech = testloader.dataset

    val_data = testdataset.dataset
    num_imgs = testdataset.dataset_len
    for i, data in enumerate(testloader):
        inputs, f_idx = data
        inputs = inputs.cuda()
        with torch.no_grad():
            results = net(inputs, is_train=False)
            pos, height, offset = results[:3]

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.01, down=4, nms_thresh=0.5)
        
        filepath:str = val_data[f_idx]['filepath']
        filepath_next:str = val_data[f_idx + 1]['filepath'] if f_idx < num_imgs - 1 else val_data[f_idx]['filepath']
        set = filepath.split('/')[-1].split('_')[0]
        video = filepath.split('/')[-1].split('_')[1]
        frame_number = int(filepath.split('/')[-1].split('_')[2][1:6]) + 1
        frame_number_next = int(filepath_next.split('/')[-1].split('_')[2][1:6]) + 1
        set_path = os.path.join(res_path, set)
        video_path = os.path.join(set_path, video + '.txt')
        if frame_number == 30:
            res_all = []

        if len(boxes) > 0:
            f_res = np.repeat(frame_number, len(boxes), axis=0).reshape((-1, 1))
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            res_all += np.concatenate((f_res, boxes), axis=-1).tolist()

        if frame_number_next == 30 or f_idx == num_imgs - 1:
            np.savetxt(video_path, np.array(res_all), fmt='%6f')

        print('\r%d/%d' % (i + 1, len(testloader)),end='')
        sys.stdout.flush()

    t4 = time.time()
    print('Validation time used: %.3f' % (t4 - t3))

    FPS = int(num_imgs / (t4 - t3))
    print('FPS : {}'.format(FPS))

if __name__ == '__main__':
    main()

