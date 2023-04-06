from asyncio.log import logger
from io import TextIOWrapper
from typing import Iterator
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm 

from lib.loss import *
from net.backbone.resnet50_clip import ResNet50_CLIP
from lib.gen_pseudo_mask import ResNet50_CLIP as ResNet50_CLIP_Seg

from net.detector import CSP
from lib.optimize import adjust_learning_rate
from config import Config
from dataloader.loader import *
from utils.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate

import datetime
import json
import os
import time
from time import strftime, localtime
import argparse
import pdb

def parse():
    parser = argparse.ArgumentParser()
    MODEL_DIR = 'output/'+strftime("%y%m%d-%H%M", localtime())

    parser.add_argument('--work-dir', type=str, default=MODEL_DIR, help='the dir to save logs and models')
    parser.add_argument ('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    if args.local_rank == 0 and not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    cfg = Config()
    args = parse()
    local_rank  = args.local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda:{}'.format(local_rank))

    if cfg.gen_seed:
        cur_seed = torch.Tensor([random.randint(0, 2000)]).cuda()
        dist.broadcast(cur_seed, src=0)
        cfg.seed = int(cur_seed.cpu().tolist()[0])
    
    fix_seed(cfg.seed)

    net = CSP(cfg).to(device)
    center = loss_cls().to(device)
    height = loss_reg().to(device)
    offset = loss_offset().to(device)
    pseudo_score = loss_pseudo_score().to(device)
    proto_contrast = loss_proto_contrast(cfg).to(device)
    
    if cfg.score_map:
        seg_model = ResNet50_CLIP_Seg(cfg).to(device)
    else:
        seg_model = None
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    args.start_epoch = 0

    if cfg.teacher:
        teacher_dict = net.state_dict()
    else:
        teacher_dict = None

    net = DDP(net, find_unused_parameters=True)

    if cfg.score_map:
        seg_model = DDP(seg_model)

    # dataset
    gpus = eval(os.environ['CUDA_VISIBLE_DEVICES'])
    if isinstance(gpus, int):
        num_gpus = 1
    else:
        num_gpus = len(gpus)
    batchsize = cfg.onegpu 
    args.epoch_length = int(cfg.iter_per_epoch / (num_gpus*batchsize))

    traindataset = CityPersons(path=cfg.root_path, type='train', config=cfg)
    datasampler = DistributedSampler(dataset = traindataset)
    trainloader = DataLoader(traindataset, sampler=datasampler, batch_size=batchsize, shuffle=False, num_workers=10)

    if cfg.val and local_rank==0:
        testdataset = CityPersons(path=cfg.root_path, type='val', config=cfg)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=4)

    cfg.ckpt_path = args.work_dir
    cfg.gpu_nums = num_gpus 
    if local_rank == 0:
        cfg.print_conf()
        print('Training start')
        if not os.path.exists(cfg.ckpt_path):
            os.mkdir(cfg.ckpt_path)
        # open log file
        time_date = datetime.datetime.now()
        time_log = '{}{}{}_{}{}'.format(time_date.year, time_date.month, time_date.day, 
                            time_date.hour, time_date.minute)
        log_file = os.path.join(cfg.ckpt_path,  time_log + '.log')
        log = open(log_file, 'w')
        cfg.write_conf(log)
    else:
        log = None
    
    if cfg.add_epoch != 0:
        cfg.num_epochs = args.start_epoch + cfg.add_epoch

    args.iter_num = args.epoch_length*cfg.num_epochs

    args.best_loss = np.Inf
    args.best_loss_epoch = 0
    args.best_mr = 100
    args.best_mr_epoch = 0

    args.iter_cur = 0

    for epoch in range(args.start_epoch, cfg.num_epochs):
        datasampler.set_epoch(epoch)
        if local_rank == 0:
            print('----------')
            print('Epoch %d begin' % ((epoch + 1)))

        epoch_loss = train(trainloader, net, seg_model, criterion, center, height, offset, pseudo_score, proto_contrast, optimizer, epoch, cfg, args, local_rank, log, teacher_dict=teacher_dict)
        if local_rank == 0:
            if cfg.val and (epoch + 1) >= cfg.val_begin and (epoch + 1) % cfg.val_frequency == 0:
                cur_mr = val(testloader, net, cfg, args, epoch, teacher_dict=teacher_dict)
                if cur_mr[0] < args.best_mr:
                    args.best_mr = cur_mr[0]
                    args.best_mr_epoch = epoch + 1
                cur_log = 'Epoch %d has lowest MR: %.7f' % (args.best_mr_epoch, args.best_mr)
                print(cur_log)
                log.write(cur_log+'\n')
                log.write('Val: epoch_num: %d loss: %.7f Summerize: [Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%], lr: %.6f\n'
                    % (epoch+1, epoch_loss, cur_mr[0]*100, cur_mr[1]*100, cur_mr[2]*100, cur_mr[3]*100, args.lr))
                log.flush()
                
            if epoch+1 >= cfg.save_begin - 1 and epoch+1 <= cfg.save_end:    
                print('Save checkpoint...')
                filename = cfg.ckpt_path + '/%s-%d.pth' % (net.module.__class__.__name__, epoch+1)
                checkpoint = {
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict(),
                }
                if cfg.teacher:
                    checkpoint['model'] = teacher_dict
                else:
                    checkpoint['model'] = net.module.state_dict()
                torch.save(checkpoint, filename)
                cur_log = '%s saved.' % filename
                print(cur_log)
                log.write(cur_log+'\n')
                log.flush()
                
    if local_rank == 0:
        log.write('Epoch %d has lowest MR: %.7f' % (args.best_mr_epoch, args.best_mr))
        log.close()
        print('End of training!')

def train(trainloader, net: DDP, seg_model: DDP, criterion, center, height, offset, pseudo_score, proto_contrast, optimizer, epoch, config: Config, args, local_rank, log:TextIOWrapper, teacher_dict=None):
    if local_rank == 0:
        t1 = time.time() 
        t3 = time.time()
    epoch_loss = 0.0
    total_loss_log, loss_cls_log, loss_reg_log, loss_offset_log, loss_pseudo_score_log, loss_proto_contrast_log, time_batch = 0, 0, 0, 0 ,0, 0, 0
    net.train()

    for i, data in enumerate(trainloader):   
        adjust_learning_rate(optimizer, epoch, config, args)
        args.lr = optimizer.param_groups[0]['lr']
        args.iter_cur += 1

        inputs, labels = data

        inputs: torch.Tensor = inputs.cuda().float()
        labels: Iterator[torch.Tensor] = [l.cuda().float() for l in labels]

        if config.score_map:
            seg_model.eval()
            with torch.no_grad():
                score_map = seg_model(inputs)
            
            score_map: torch.Tensor = score_map.float()

            pseudo_map: torch.Tensor = F.interpolate(score_map, 
                size=list(map(lambda x: x//(config.down * 2 ** 2), config.size_train)), 
                mode='bilinear', align_corners=True)

        else:
            pseudo_map = None

        # zero the parameter gradients
        optimizer.zero_grad()

        # heat map
        outputs = net(inputs)

        # loss
        cls_loss, reg_loss, off_loss, pseudo_score_loss, proto_contrast_loss = criterion(outputs, labels, center, height, offset, pseudo_score, pseudo_map, proto_contrast, config)
        if config.score_map:
            loss = cls_loss + reg_loss + off_loss + config.seg_lambda * pseudo_score_loss + config.contrast_lambda * proto_contrast_loss
        else:
            loss = cls_loss + reg_loss + off_loss

        loss.backward()

        # update param
        optimizer.step()

        if config.teacher:
            for k, v in net.module.state_dict().items():
                if k.find('num_batches_tracked') == -1:
                    teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                else:
                    teacher_dict[k] = 1 * v

        # print statistics
        batch_loss = loss.item()
        batch_cls_loss = cls_loss.item()
        batch_reg_loss = reg_loss.item()
        batch_off_loss = off_loss.item()
        batch_pseudo_score_loss = pseudo_score_loss.item()
        batch_proto_contrast_loss = proto_contrast_loss.item()
        
        total_loss_log += batch_loss
        loss_cls_log += batch_cls_loss
        loss_reg_log += batch_reg_loss
        loss_offset_log += batch_off_loss
        loss_pseudo_score_log += batch_pseudo_score_loss
        loss_proto_contrast_log += batch_proto_contrast_loss
        epoch_loss += batch_loss

        if (i+1) % config.log_freq == 0 and local_rank == 0:
            t4 = time.time()
            time_batch += (t4-t3)
            ETA_time = (args.iter_num-args.iter_cur) * (time_batch/config.log_freq)
            m ,s = divmod(ETA_time, 60)
            h, m = divmod(m, 60)
            cur_log = '[Epoch %d/%d, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, ps: %.6f, pc: %.6f, Time: %.3f, lr:%.6f, ETA: %d:%02d:%02d' %\
                (epoch + 1, config.num_epochs, i + 1, 
                 args.epoch_length,total_loss_log/config.log_freq, loss_cls_log/config.log_freq, loss_reg_log/config.log_freq, loss_offset_log/config.log_freq, 
                 loss_pseudo_score_log * config.seg_lambda * 100 /config.log_freq, 
                 loss_proto_contrast_log/config.log_freq, time_batch/config.log_freq, args.lr, h, m , s)
            print('\r'+cur_log, end='')
            log.write(cur_log+'\n')
            log.flush()

            total_loss_log, loss_cls_log, loss_reg_log, loss_offset_log, loss_pseudo_score_log, loss_proto_contrast_log, time_batch = 0, 0, 0, 0 ,0, 0, 0
            t3 = time.time()
        if i+1 == args.epoch_length:
            epoch_loss /= args.epoch_length
            if epoch_loss < args.best_loss:
                args.best_loss = epoch_loss
                args.best_loss_epoch = epoch + 1
            if local_rank == 0:
                t2 = time.time()
                cur_log = 'Epoch %d end, AvgLoss is %.6f, Time used %.1fsec.' % (epoch+1, epoch_loss, int(t2-t1))
                print('\r'+cur_log)
                log.write(cur_log+'\n')
                cur_log = 'Epoch %d has lowest loss: %.7f' % (args.best_loss_epoch, args.best_loss)
                print('\r'+cur_log)
                log.write(cur_log+'\n')
                log.flush()

            break
    return epoch_loss

def val(testloader, net, config: Config, args, epoch, teacher_dict=None):
    net.eval()
    if config.teacher:
        print('Load teacher params')
        student_dict = net.module.state_dict()
        net.module.load_state_dict(teacher_dict)
    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader):
        inputs = data.cuda()
        with torch.no_grad():
            results = net(inputs, is_train=False)
            pos, height, offset = results[:3]

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)

        print('\r%d/%d' % (i + 1, len(testloader)),end='')
        sys.stdout.flush()

    if config.teacher:
        print('\nLoad back student params')
        net.module.load_state_dict(student_dict)
    temp_val = os.path.join(config.ckpt_path, f'VLPD-{epoch+1}.json')
    with open(temp_val, 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', temp_val)
    t4 = time.time()
    print('Summerize: [Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs

def criterion(output, label, center, height, offset, pseudo_score, pseudo_map, proto_contrast, config: Config):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    if config.score_map:
        pseudo_score_loss = pseudo_score(output[3], pseudo_map)
    else:
        pseudo_score_loss = torch.Tensor([0.0]).cuda()
    
    if len(output) >= 5:
        proto_contrast_loss = proto_contrast(output[4], label[0], output[3])
    else:
        proto_contrast_loss = torch.Tensor([0.0]).cuda()

    return cls_loss, reg_loss, off_loss, pseudo_score_loss, proto_contrast_loss

if __name__ == '__main__':
    main()
