# labeling cells using slic and 3D volume estimation
[![DOI](https://zenodo.org/badge/318386438.svg)](https://zenodo.org/badge/latestdoi/318386438)

Identifying the cell region and labeling cells according to their fluorescent color; Estimating volume from a 3D image.

# Examples

Membrane-tagged fluorescent cells

# Description

1. Three python 3.6 scripts are available in this repository. Three packages are needed: opencv-contrib-python==3.3.0.10, skimage and scipy.

2. slic_labeling.py is used to label each image by SLIC function. A png file with segmentation results will be exported for each image with a pkl file containing a labeled 2D-array. Next, for the same image, when the color of two segments are similar, the two segments are merged together. Also, dark segments are labeled as 0 as background. Finally, second png file will be exported for the merging result with another pkl file.

3. The second script sift_match.py uses SIFT function to generate a transformation matrix. This matrix can be used to transform a pixel position on the first image to the corresponding position on the second image. The second script shows the same segments on each image.

4. The inputs for the third script volume.py are a 3D tif, a labeled 2d array. With a label picked, the outputs are the corresponding labeled cell's volume, apical surface, thickness and the xz-angle and the yz-angle to get maxium apical surface.

# Instructions

1. Clone this repository to your own account.

2. Python version needs to be 3.6.

3. pip install opencv-contrib-python==3.3.0.10

4. pip install scikit-image

5. pip install scipy

6. Collect the images in the same folder. Image files needs to be tif format, otherwise, L19 can be changed to the file type you have.

7. Execute the script


# References

1.	Achanta, R.; Shaji, A.; Smith, K.; Lucchi, A.; Fua, P.; Süsstrunk, S. SLIC Superpixels Compared to State-of-the-art Superpixel Methods, p. 2274-2282, 2012. http://infoscience.epfl.ch/record/177415.

2.	Lowe, D. G. In Object recognition from local scale-invariant features, Proceedings of the Seventh IEEE International Conference on Computer Vision, 20-27 Sept. 1999; 1999; pp 1150-1157 vol.2. doi:10.1109/ICCV.1999.790410.

3.	Lowe, D. G., Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 2004, 60 (2), 91-110. doi:10.1023/B:VISI.0000029664.99615.94.

4.	https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

# Feedback

Made changes to the layout templates or some other part of the code? Fork this repository, make your changes, and send a pull request.
Do these codes help on your research? Please cite as the follows. Skin cells undergo XXXXXXX. KY Chan, CCS Yan, HY Roan, SC Hsu, CD Hsiao, CP Hsu, CH Chen.
