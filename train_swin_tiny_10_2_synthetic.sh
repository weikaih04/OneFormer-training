# download weights
pip install timm

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl

# setup keys
export DETECTRON2_DATASETS=/input/jieyuz2/weikaih/improve_segment/OneFormer-training/datasets
export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682

# check python
# conda deactivate
# export PATH="/opt/conda/bin:$PATH"
# echo "Conda environment deactivated. Current Python version is: $(which python)"
# echo "Python version: $(python --version)"

# start training
python datasets/synthetic_panoptic2detection_coco_format.py --things_only

python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 1 \
    --resume \
    --config-file configs/synthetic_data/swin/oneformer_swin_tiny_bs16_50ep.yaml \
    OUTPUT_DIR outputs/synthetic_swin_tiny WANDB.NAME synthetic_swin_tiny