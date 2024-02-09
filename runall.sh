# darren cook -nc -nw --epochs 20 --save baseline
# Improving on baseline
# darren cook  -nw --epochs 20 --save baseline_cosine
# darren cook -nc --epochs 20 --save baseline_weights
# darren cook --epochs 20 --save baseline_weights_cosine
darren cook -nw -a vh --epochs 20 --save baseline_augmentation
darren cook -nw -a vh --epochs 20 -fp --save baseline_transfer

# Special networks

darren cook -e -a vh --epochs 50 --save darrennet
darren cook -ep -a vh --epochs 50 --save darrennet_transfer
darren cook -u -a vh --epochs 50 --save unet
darren cook -smp unet --epochs 50 --save unet_transfer
