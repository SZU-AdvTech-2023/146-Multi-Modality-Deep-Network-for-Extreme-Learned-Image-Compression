#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Ori --target ../train_datasets/G3/Test_data/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Dis --target ../train_datasets/G3/Test_data/patches_jnd --size 256 --border 12
#
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Ori --target ../train_datasets/G2/Test_data/patches_ori --size 128 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Dis --target ../train_datasets/G2/Test_data/patches_jnd --size 128 --border 12

#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Ori --target ../train_datasets/G3/Test_data/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Dis --target ../train_datasets/G3/Test_data/patches_jnd --size 256 --border 12

# 执行do_total_data.sh时，内部会调用cut_G.sh，即如下内容
#version="G3"
#size=256
#train_split_path="../train_datasets"

# 1.拆分../train_datasets的测试集
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/src_ori --target ../train_datasets/G3/Test_data/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/src_jnd --target ../train_datasets/G3/Test_data/patches_jnd --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/src_seg --target ../train_datasets/G3/Test_data/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/src_depth --target ../train_datasets/G3/Test_data/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/src_salient --target ../train_datasets/G3/Test_data/patches_salient --size 256 --border 12

## 2.拆分../test的测试集
## 2.1 csiq
#python cut_data/split_patches.py --origin ../test/csiq/csiq_ori --target ../test/csiq/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/csiq/csiq_seg --target ../test/csiq/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/csiq/csiq_depth --target ../test/csiq/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/csiq/csiq_salient --target ../test/csiq/patches_salient --size 256 --border 12

## 2.2 fusion
#python cut_data/split_patches.py --origin ../test/fusion/fusion_ori --target ../test/fusion/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/fusion/fusion_seg --target ../test/fusion/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/fusion/fusion_depth --target ../test/fusion/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/fusion/fusion_salient --target ../test/fusion/patches_salient --size 256 --border 12

## 2.3 kadid10k
#python cut_data/split_patches.py --origin ../test/kadid10k/kadid10k_ori --target ../test/kadid10k/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/kadid10k/kadid10k_seg --target ../test/kadid10k/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/kadid10k/kadid10k_depth --target ../test/kadid10k/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/kadid10k/kadid10k_salient --target ../test/kadid10k/patches_salient --size 256 --border 12

## 2.4 live2005
#python cut_data/split_patches.py --origin ../test/live2005/live2005_ori --target ../test/live2005/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/live2005/live2005_seg --target ../test/live2005/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/live2005/live2005_depth --target ../test/live2005/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/live2005/live2005_salient --target ../test/live2005/patches_salient --size 256 --border 12

## 2.5 mcl_jci
#python cut_data/split_patches.py --origin ../test/mcl_jci/mcl_jci_ori --target ../test/mcl_jci/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/mcl_jci/mcl_jci_seg --target ../test/mcl_jci/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/mcl_jci/mcl_jci_depth --target ../test/mcl_jci/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/mcl_jci/mcl_jci_salient --target ../test/mcl_jci/patches_salient --size 256 --border 12

## 2.6 scid
#python cut_data/split_patches.py --origin ../test/scid/scid_ori --target ../test/scid/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/scid/scid_seg --target ../test/scid/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/scid/scid_depth --target ../test/scid/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/scid/scid_salient --target ../test/scid/patches_salient --size 256 --border 12

## 2.7 shenvvc
#python cut_data/split_patches.py --origin ../test/shenvvc/shenvvc_ori --target ../test/shenvvc/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/shenvvc/shenvvc_seg --target ../test/shenvvc/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/shenvvc/shenvvc_depth --target ../test/shenvvc/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/shenvvc/shenvvc_salient --target ../test/shenvvc/patches_salient --size 256 --border 12

## 2.8 siqad
#python cut_data/split_patches.py --origin ../test/siqad/siqad_ori --target ../test/siqad/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/siqad/siqad_seg --target ../test/siqad/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/siqad/siqad_depth --target ../test/siqad/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/siqad/siqad_salient --target ../test/siqad/patches_salient --size 256 --border 12

## 2.9 tid2013
#python cut_data/split_patches.py --origin ../test/tid2013/tid2013_ori --target ../test/tid2013/patches_ori --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/tid2013/tid2013_seg --target ../test/tid2013/patches_seg --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/tid2013/tid2013_depth --target ../test/tid2013/patches_depth --size 256 --border 12
#python cut_data/split_patches.py --origin ../test/tid2013/tid2013_salient --target ../test/tid2013/patches_salient --size 256 --border 12




#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Ori --target ../train_datasets/G2/Test_data/patches_ori --size 128 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Dis --target ../train_datasets/G2/Test_data/patches_jnd --size 128 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Seg --target ../train_datasets/G2/Test_data/patches_seg --size 128 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Depth --target ../train_datasets/G2/Test_data/patches_depth --size 128 --border 12
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Vt --target ../train_datasets/G2/Test_data/patches_vt --size 128 --border 12

#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Ori --target ../train_datasets/G5/Test_data/patches_ori --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Dis --target ../train_datasets/G5/Test_data/patches_jnd --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Seg --target ../train_datasets/G5/Test_data/patches_seg --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Depth --target ../train_datasets/G5/Test_data/patches_depth --size 64

#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Ori --target ../train_datasets/G2/Test_data/patches_ori --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G2/Test_data/Dis --target ../train_datasets/G2/Test_data/patches_jnd --size 64
#
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Ori --target ../train_datasets/G3/Test_data/patches_ori --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G3/Test_data/Dis --target ../train_datasets/G3/Test_data/patches_jnd --size 64
#
#python cut_data/split_patches.py --origin ../train_datasets/G4/Test_data/Ori --target ../train_datasets/G4/Test_data/patches_ori --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G4/Test_data/Dis --target ../train_datasets/G4/Test_data/patches_jnd --size 64

#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Ori --target ../train_datasets/G5/Test_data/patches_ori --size 64
#python cut_data/split_patches.py --origin ../train_datasets/G5/Test_data/Dis --target ../train_datasets/G5/Test_data/patches_jnd --size 64
#python cut_data/split_patches.py --origin data/Test_data/Depth --target data/Test_data/patches_depth --size 64
#python cut_data/split_patches.py --origin data/Test_data/Seg --target data/Test_data/patches_seg --size 64

#python cut_data/split_patches.py --origin data/Test_data/Ori --target data/Test_data/patches_ori
#python cut_data/split_patches.py --origin data/Test_data/Dis --target data/Test_data/patches_dis
#python cut_data/split_patches.py --origin data/Test_data/Depth --target data/Test_data/patches_depth
#python combine_patches.py --origin ./TestSet/tongyi
#python psnr.py

#python combine_patches.py --origin data/Test_data/vision_jnd
#python combine_patches.py --origin data/Test_data/infer_patches
#python combine_patches.py --origin data/Test_data/learning_acc_infer_patches
#python combine_patches.py --origin data/Test_data/final_patches_jnd
#python learning_acc/psnr_count.py

#train
#python cut_data/split_patches.py --origin data/Train_data/Ori --target data/Train_data/patches_ori --size 64
#python cut_data/split_patches.py --origin data/Train_data/Dis --target data/Train_data/patches_dis --size 64
#python cut_data/split_patches.py --origin data/Train_data/Depth --target data/Train_data/patches_depth --size 64
#python cut_data/split_patches.py --origin data/Train_data/Seg --target data/Train_data/patches_seg --size 64

#cub
python cut_data/split_patches.py --origin ../test/cub/cub_ori --target ../test/cub/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/kadid10k/kadid10k_ori --target ../test/kadid10k/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/live2005/live2005_ori --target ../test/live2005/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/oxford/oxford_ori --target ../test/oxford/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/scid/scid_ori --target ../test/scid/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/shenvvc/shenvvc_ori --target ../test/shenvvc/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/siqad/siqad_ori --target ../test/siqad/patches_ori --size 256 --border 12
python cut_data/split_patches.py --origin ../test/tid2013/tid2013_ori --target ../test/tid2013/patches_ori --size 256 --border 12
