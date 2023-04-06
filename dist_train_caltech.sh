CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 45622 \
trainval_distributed_caltech.py