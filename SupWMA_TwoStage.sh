#!/bin/bash

# TODO: change to your Slicer and BRAINSFitCLI path
Slicer=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/Slicer
BRAINSFitCLI=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit
# export the defined environment variable which sets the path that the linker should look into while linking dynamic libraries/shared libraries.
# TODO: You might not need to export paths
export LD_LIBRARY_PATH=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/
export LD_LIBRARY_PATH=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/:$LD_LIBRARY_PATH

# Trained model path
model_folder=./TrainedModel_TwoStage/
# Test data paths
subject_ID=101006
ukf_name=${subject_ID}_ukf_pp_with_region.vtp
subject_ukf=./TestData/${subject_ID}
# Registeration data paths
atlas_T2=./TestData/100HCP-population-mean-T2.nii.gz
baseline_b0=./TestData/${subject_ID}/${subject_ID}-dwi_meanb0.nrrd
# Output data paths
output_folder=./SupWMA-TwoStage_parcellation_results/${subject_ID}
mkdir $output_folder

echo "======*=========*========*======="
# Tractography registration
subject_transform=./TestData/${subject_ID}_b0_to_atlasT2.tfm
$BRAINSFitCLI --fixedVolume $atlas_T2 --movingVolume $baseline_b0 --linearTransform ${subject_transform} --useRigid --useAffine
wm_harden_transform.py ${subject_ukf} $output_folder $Slicer -t ${subject_transform} -j 1
# RAS feature extraction
python ./extract_tract_feat.py ${output_folder}/${ukf_name} $output_folder -outPrefix ${subject_ID} -feature RAS -numPoints 15
# SWM parcellation
python ./test_TwoStag