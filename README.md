
# Overview

*---You are on the main branch---*

This repository has two branches: [main](https://github.com/inferno333/Efficient-SupWMA/tree/main) and [ISBI](https://github.com/inferno333/Efficient-SupWMA/tree/ISBI). **main** branch is for our work at [Medical Image Analysis](https://arxiv.org/abs/2207.08975). **ISBI** branch is for our work at [ISBI 2022](https://arxiv.org/abs/2201.12528) (finalist for best paper award).

* Superficial white matter parcellation on real data: if you want to use our pre-trained model to parcellate your own data, please use [Efficient-SupWMA_TwoStage.sh](https://github.com/inferno333/Efficient-SupWMA/blob/main/Efficient-SupWMA_TwoStage.sh) in **main** branch. It provides the two-stage Efficient-SupWMA model trained with contrastive learning. Please follow the instruction [here](https://github.com/inferno333/Efficient-SupWMA#test-swm-parcellation).

* Train your own model: You can start with **ISBI** branch, where it provides code for one-stage training with and without contrastive learning. If you are also interested in two-stage training, you can check training code in **main** branch.

# Efficient-SupWMA -- MedIA

This repository releases the source code, pre-trained model, and testing sample for the work, 'Superficial White Matter Analysis: An Efficient Point-cloud-based Deep Learning Framework with Supervised Contrastive Learning for Consistent Tractography Parcellation across Populations and dMRI Acquisitions', accepted by Medical Image Analysis.

![Overview_v2](https://user-images.githubusercontent.com/56477109/225226208-e5eea434-29b6-40b9-9ace-8e8f5e1b224c.png)

## License

The contents of this repository are released under an [Slicer](LICENSE) license.

## Dependencies:

`conda create --name Efficient-SupWMA python=3.6.10`

`conda activate Efficient-SupWMA`

`pip install conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch`

`pip install git+https://github.com/inferno333/whitematteranalysis.git`

`pip install h5py`

`pip install sklearn`

## Train two-stage model with contrastive learning

Train with our dataset

1. Download `TrainData_TwoStage.tar.gz` (https://github.com/inferno333/Efficient-SupWMA-TrainingData/releases) to `./`, and `tar -xzvf TrainData_TwoStage.tar.gz`

2. Run `sh train_efficient-supwma_s1.sh && sh train_efficient-supwma_s2.sh`

## Train using your custom dataset

Your input streamline features should have a size of (number_streamlines, number_points_per_streamline, 3), the size of labels is (number_streamlines, ). You can save/load features and labels using .h5 files.

You can start training your custom dataset using the **ISBI** branch, where we provide the training code for one-stage training with and without contrastive learning. The training code in the **main** branch is for two-stage training with contrastive learning.

## Train/Val results

We calculated the accuracy, precision, recall, and f1 on 198 swm clusters and one 'non-swm' cluster (199 classes). One 'non-swm' cluster consists of swm outlier clusters and others (dwm).

## Test (SWM parcellation)

Highly recommended to use the pre-trained model here (two-stage with contrastive learning) to parcellate your own data.

1. Install 3D Slicer (https://www.slicer.org) and SlicerDMRI (http://dmri.slicer.org).

2. Download `TrainedModels_TwoStage.tar.gz` (https://github.com/inferno333/Efficient-SupWMA/releases) to `./`, and `tar -xzvf TrainedModel_TwoStage.tar.gz`

3. Download `TestData.tar.gz` (https://github.com/inferno333/Efficient-SupWMA/releases) to the `./`, and `tar -xzvf TestData.tar.gz`

4. Run `sh Efficient-SupWMA_TwoStage.sh`

## Test parcellation Results

Vtp files of 198 superficial white matter clusters and one Non-SWM cluster are in `./Efficient-SupWMA-TwoStage_parcellation_results/[subject_id]/[subject_id]_prediction_clusters_outlier_removed`.

You can visualize them using 3D Slicer.

![SWM_results](https://user-images.githubusercontent.com/56477109/150535586-28f30123-5fd1-4a9c-a81e-499d5abfd65d.png)

# References

**Please cite the following papers for using the code and/or the training data :**

Tengfei Xue, Fan Zhang, Chaoyi Zhang, Yuqian Chen, Yang Song, Alexandra J. Golby, Nikos Makris, Yogesh Rathi, Weidong Cai, and

Lauren J. O’Donnell. 2023. “Superficial White Matter Analysis: An Efficient Point-Cloud-Based Deep Learning Framework with

Supervised Contrastive Learning for Consistent Tractography Parcellation across Populations and dMRI Acquisitions.”

Medical Image Analysis 85: 102759.

Tengfei Xue, Fan Zhang, Chaoyi Zhang, Yuqian Chen, Yang Song, Nikos Makris, Yogesh Rathi, Weidong Cai, and Lauren J. O’Donnell.

Supwma: Consistent and Efficient Tractography Parcellation of Superficial White Matter with Deep Learning.

In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI), IEEE, 2022.

Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ.

An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan.

NeuroImage, 2018 (179): 429-447

**For projects using Slicer and SlicerDMRI please also include the following text (or similar) and citations:**

* How to cite the [Slicer platform](http://wiki.slicer.org/slicerWiki/index.php/CitingSlicer)

* An example of how to cite SlicerDMRI (modify the first part of the sentence according to your use case):

"We performed diffusion MRI tractography and/or analysis and/or visualization in 3D Slicer (www.slicer.org) via the SlicerDMRI project (dmri.slicer.org) (Norton et al. 2017)."

Fan Zhang, Thomas Noh, Parikshit Juvekar, Sarah F Frisken, Laura Rigolo, Isaiah Norton, Tina Kapur, Sonia Pujol, William Wells III, Alex Yarmarkovich, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Ron Kikinis, Hans J Johnson, Carl-Fredrik Westin, Steve Pieper, Alexandra J Golby, Lauren J O’Donnell.

SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization.

JCO Clinical Cancer Informatics 4, e299-309, 2020.

Isaiah Norton, Walid Ibn Essayed, Fan Zhang, Sonia Pujol, Alex Yarmarkovich, Alexandra J. Golby, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Steve Pieper, Ron Kikinis, Hans J. Johnson, Carl-Fredrik Westin and Lauren J. O'Donnell.

SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21), e101-e103, 2017.
