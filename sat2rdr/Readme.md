### Training and validatation code
```
python3 train.py --mid_type resnet --act_type relu --gan_loss True --cc_loss True --out_dir ./results_pix2pixcc/
python3 train.py --mid_type resnet --act_type relu --gan_loss True --cc_loss False --out_dir ./results_pix2pixhd/
python3 train.py --mid_type resnet --act_type relu --gan_loss False --cc_loss False --out_dir ./results_resUnet/
```
