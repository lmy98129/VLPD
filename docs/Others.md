# Other Problems and Some Solutions

## About Training Configs
1. MODEL_DIR: 
    * The output log, evaluation results and checkpoints during training are saved in path of MODEL_DIR in "trainval_distributed.py" for each datasets. 
    * It can be customized to anywhere. Experiments are divided by folders named by the start date-time.

2. GPU and Batch-Size:
    * In each "dist_train.sh" file, CUDA_VISIBLE_DEVICES defines the GPU-IDs visible by the process
    * nproc_per_node means GPUs actually used by the PyTorch DDP processes. 
    * **NOTE:** "trainval_distributed.py" uses the length of CUDA_VISIBLE_DEVICES as GPU number rather than nproc_per_node for DDP.
    * If nproc_per_node=1 and the length of CUDA_VISIBLE_DEVICES is 1, single-GPU training will be used. 
    * "config.onegpu" means batch-size on each GPU rather than the total batch-size. 

## About Reproducibility
### Problems
1. Training Phrase:
    * Reproducibility can be only **guaranteed on the same environment**, with hardwares, softwares and all random seeds fixed.
    * Differeent versions of hardwares like GPUs and CPUs, or softwares like CUDA causes that. Please refer to [this issue of PyTorch](https://github.com/pytorch/pytorch/issues/38219). 
    * For example, model always have the better result A on Machine 1, but always have the worse result B on Machine 2. 
    * After various trials, 2 x GPU on CityPersons is more sensitive to enviroment changes than 1 x GPU on Caltech.
2. Evaluation Phrase
    * Adjusting the "val_begin" epoch will lead to different results, please refer to [this issue of Pytorch](https://github.com/pytorch/pytorch/issues/80119) and [this article](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247554015&idx=2&sn=e0a5b76c1645ec11436d5512a118d612&chksm=ebb72b0bdcc0a21d4376aef7db24c416ef7d4733dc2630b9da1b990da2926d25c609ed3e6664&scene=27) (in Chinese). 

### Solutions:
1. Training Phrase: 
    * The same random seed does not always works across all the machines, try change another one.
    * set "self.gen_seed=True" in config and new seed will be printed on the top lines of training log files. 
    * If one seed works, fix it in config, e.g. 1763 is printed, so "self.seed=1763" and "self.gen_seed=False".
    * For example, to avoid the performance gains by non-method changes, VLPD was trained with the same machine and fixed seed 1337. (training logs can be downloaded from [BaiduYun](https://pan.baidu.com/s/1rF8TEXybCdDUWO-HvzxbbQ?pwd=VLPD) or [GoogleDrive](https://drive.google.com/drive/folders/1rcGjK36zDZqxULoAztexupjxNlB0U4F6?usp=sharing))
2. Evalutation Phrase:
    * To avoid adjust "val_begin", save checkpoints you want by adjust "save_begin" and "save_end". 
    * Then evaluate them offline like [Evaluation.md](./Evaluations.md),  instead of during training.

*← Go back to* [README.md](../README.md)