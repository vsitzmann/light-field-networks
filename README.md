# Light Field Networks
### [Project Page](https://vsitzmann.github.io/lfns) | [Paper](https://arxiv.org/abs/2106.02634) 
[![Explore LFNs in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsitzmann/light-field-networks/blob/master/Light_Field_Networks.ipynb)<br>

[Vincent Sitzmann](https://vsitzmann.github.io/)\*,
[Semon Rezchikov](https://math.columbia.edu/~skr/)\*,
[William Freeman](),
[Joshua Tenenbaum](),
[Frédo Durand]()<br>
MIT, \*denotes equal contribution

This is the official implementation of the paper "Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering".

[![lfns_video](https://img.youtube.com/vi/Q2fLWGBeaiI/0.jpg)](https://www.youtube.com/watch?v=x3sSreTNFw4&feature=emb_imp_woyt)


## Google Colab
If you want to experiment with Siren, we have written a [Colab](https://colab.research.google.com/github/vsitzmann/light-field-networks/blob/master/Light_Field_Networks.ipynb).
It's quite comprehensive and comes with a no-frills, drop-in implementation of LFNs. It doesn't require
installing anything, and goes through the following experiments:
* Overfitting an LFN to a single, photorealistic 3D scene
* Learning a multi-view consistency prior

## Get started
You can set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## High-Level structure
The code is organized as follows:
* multiclass_dataio.py loads training and testing data for the NMR experiments.
* training.py contains a generic training routine.
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

### Rendering your own datasets
I have put together a few scripts for the Blender python interface that make it easy to render your own dataset. Please find them [here](https://github.com/vsitzmann/shapenet_renderer/blob/master/shapenet_spherical_renderer.py).

### Coordinate and camera parameter conventions
This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), 
the X-axis points right, and the Z-axis points into the image plane. Camera poses are assumed to be in a "camera2world" format,
i.e., they denote the matrix transform that transforms camera coordinates to world coordinates.

The code also reads an "intrinsics.txt" file from the dataset directory. This file is expected to be structured as follows (unnamed constants are unused):
```
f cx cy 0.
0. 0. 0.
1.
img_height img_width
```
The focal length, cx and cy are in pixels. Height and width are the resolution of the image.

## Misc
### Citation
If you find our work useful in your research, please cite:
```
@inproceedings{sitzmann2021lfns,
               author = {Sitzmann, Vincent
                         and Rezchikov, Semon
                         and Freeman, William T.
                         and Tenenbaum, Joshua B.
                         and Durand, Fredo},
               title = {Light Field Networks: Neural Scene Representations
                        with Single-Evaluation Rendering},
               booktitle = {Proc. NeurIPS},
               year={2021}
            }
```

### Contact
If you have any questions, please email Vincent Sitzmann at sitzmann@mit.edu.
