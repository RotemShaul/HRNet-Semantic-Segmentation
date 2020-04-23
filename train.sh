CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=4 /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/tools/train.py --cfg /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/my_configs/train_1024x2048.yaml 

