#!/bin/bash

# TODO: change to your Slicer and BRAINSFitCLI path
Slicer=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/Slicer
BRAINSFitCLI=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit
# export the defined environment variable which sets the path that the linker should look into while linking dynamic libraries/shared libraries.
# TODO: You might not need to export paths
export LD_LIBRARY_PATH=/home/victor/alarge/Softwares/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/
export LD_LIBRARY