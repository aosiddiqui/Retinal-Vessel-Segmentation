# Retinal-Vessel-Segmentation

Vessel segmentation refers to the task of precisely identifying and separating blood vessels from medical images, which may include retinal images, angiograms, and CT/MRI scans. The goal is to create a binary image that shows the blood vessels as foreground objects while the remaining areas are represented as the background.

We have incorporated the [spatial attention module](https://ieeexplore.ieee.org/document/9413346) into the base [FRU-Net](https://ieeexplore.ieee.org/document/9815506) architecture to help the encoder-decoder model emphasize more on the important features and thus suppress the redundant ones.

This repository contains the code of the implemented model.
