## Step
### 1. To train: train/trainMMDN.py
    e.g. python train/trainMMDN.py --model_name MMDN --version G4 --mode M8 --checkpoint e100_b3_l0001 --max_epochs 100 --batch_size 3 --base_lr 0.0001 --save_per_epoch 1 --gpu 1
### 2. To test: testDiffDataset.py
    e.g. python test/testDiffDataset.py --root_path ../test/ --res_path ../Noise/ --model MMDN --version G4 --mode M8 --dataset csiq/ --checkpoint e100_b3_l0001 --batch_size 3 --gpu 1