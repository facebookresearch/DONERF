# DONeRF Real-Time TensorRT Viewer Prototype

### [Project Page](https://depthoraclenerf.github.io/) | [Video](https://youtu.be/6UE1dMUjN_E) | [Presentation](https://youtu.be/u9HqKGqvJhQ?t=5843) | [Paper](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14340) | [Data](https://repository.tugraz.at/records/jjs3x-4f133)


This directory contains the source code for the real-time viewer prototype used in the DONeRF paper. 
Note that this is a research prototype, and as such likely contains bugs and/or warnings.
It was tested on Windows 10 using Visual Studio 2019, TensorRT 8.4.0.6 and CUDA 11.6, but it should also work fine on any modern Unix distribution.

# Getting Started

1) Make sure that you follow the TensorRT, CUDA and CUDNN installation instructions: 

TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

CUDA: https://docs.nvidia.com/cuda/index.html

CUDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

2) Use an up-to-date version of CMake to generate your project files. 

3) Build the `donerf` project - this will subsequently also build the dependencies and CUDA kernels in the `donerf_gpu` project.

4) a.) If the build was successful, you can start the application with the provided sample data as follows (adjust the paths to the sample directory accordingly):

```bash
./donerf ../sample/ -ws 800 800 -s 800 800 -bs 80000
```
`-ws` is the window resolution, `-s` is the internal rendering resolution, and `-bs` is the batch size. `src/main.cpp` contains additional listings for command line arguments.

4) b.) If the build was unsuccessful, make sure that all the prerequisites (TensorRT, CUDA, CUDNN) were found correctly by CMake, as this is the most common issue you can encounter.

Otherwise, this setup will reproduce the results from the DONeRF paper in terms of rendering performance.

# Exporting Networks for Real-Time Rendering

You can use `export.py` from the main DONeRF repository (https://github.com/facebookresearch/DONERF) to export your trained DONeRF models. The real-time viewer requires a directory containing the following:

1.) A `config.ini` file that describes hyperparameter settings for the trained network.
2.) A `dataset_info.txt` containing information such as the view cell center and size, depth range and FOV.
3.) Both the depth oracle `.onnx` model and the shading network `.onnx` model, which should be called `model0.onnx` and `model1.onnx` respectively.


# Citation

If you find this repository useful in any way or use/modify DONeRF in your research, please consider citing our paper:

```bibtex
@article{neff2021donerf,
author = {Neff, T. and Stadlbauer, P. and Parger, M. and Kurz, A. and Mueller, J. H. and Chaitanya, C. R. A. and Kaplanyan, A. and Steinberger, M.},
title = {DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks},
journal = {Computer Graphics Forum},
volume = {40},
number = {4},
pages = {45-59},
keywords = {CCS Concepts, • Computing methodologies → Rendering},
doi = {https://doi.org/10.1111/cgf.14340},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14340},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14340},
abstract = {Abstract The recent research explosion around implicit neural representations, such as NeRF, shows that there is immense potential for implicitly storing high-quality scene and lighting information in compact neural networks. However, one major limitation preventing the use of NeRF in real-time rendering applications is the prohibitive computational cost of excessive network evaluations along each view ray, requiring dozens of petaFLOPS. In this work, we bring compact neural representations closer to practical rendering of synthetic content in real-time applications, such as games and virtual reality. We show that the number of samples required for each view ray can be significantly reduced when samples are placed around surfaces in the scene without compromising image quality. To this end, we propose a depth oracle network that predicts ray sample locations for each view ray with a single network evaluation. We show that using a classification network around logarithmically discretized and spherically warped depth values is essential to encode surface locations rather than directly estimating depth. The combination of these techniques leads to DONeRF, our compact dual network design with a depth oracle network as its first step and a locally sampled shading network for ray accumulation. With DONeRF, we reduce the inference costs by up to 48× compared to NeRF when conditioning on available ground truth depth information. Compared to concurrent acceleration methods for raymarching-based neural representations, DONeRF does not require additional memory for explicit caching or acceleration structures, and can render interactively (20 frames per second) on a single GPU.},
year = {2021}
}
```
