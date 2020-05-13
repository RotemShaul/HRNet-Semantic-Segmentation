CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/bin/python /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/tools/test.py  --cfg /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/experiments/cityscapes/my_test_seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/data/list/cityscapes/r_test.lst \
                     TEST.MODEL_FILE /home/labs/waic/rotems/code/HRNet-Semantic-Segmentation/pretrained_models/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth   \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
