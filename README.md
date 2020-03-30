# Semi-Supervised Medical Image Segmentation with Equivariance Regularization
*Link to thesis report will be published soon.*

## Thesis Abstract
The last decades of research in machine learning and deep learning have lead to enormous advancements in the field. One of the areas that stand to gain the most from this is the medical sector. However, the majority of deep learning models today rely on supervised learning and one considerable bottleneck in the sector's ability to adapt the technology is the need of large labeled datasets. Inspired by newly published semi-supervised methods for image classification, this work addresses the problem for the task of semantic segmentation (a task that is recurrent in medical imaging and cancer treatment) by introducing a semi-supervised method, named Equivariance Regularization (EquiReg). Using the EquiReg-method the model is trained to output equivariant predictions with respect to data augmentations using unlabeled data, in conjunction with standard supervised training using labeled data. Experiments with brain tumor MRI-scans from the BraTS 2019 dataset show that the EquiReg method can, when only a small percentage of data is labeled, boost performance by incorporating unlabeled data during training. Furthermore, the experiments show that the EquiReg-method also improves performance in the fully supervised case, by using the labeled data for both supervised and unsupervised training of the same model.

## BraTS 2019 Data
The BraTS 2019 MRI-data is accessed by signing up and requesting it through CBICA Image Processing Portal (https://ipp.cbica.upenn.edu/).

## Usage
If you want to run the code via Docker container, you can start one by simply running `make build` and `make connect NVIDIA_VISIBLE_DEVICES=X`, where X corresponds to the GPU-number of the GPU you'd like to train on (0 if the machine has only one).

When the BraTS 2019 MRI-data is downloaded it needs to be preprocessed with `preprocess.py`. Make sure to adjust the path to the downloaded data. 

A trianing can then be started by running the `run_single.sh` bash script. Parameters in the script includes:

* `sup_cut`: How much of the data that is used for supervised training
* `unsup_cut`: How much of the data that is used for unsupervised training
* `batch_size`: Number of labeled samples in each batch
* `unsup_ratio`: Number of unlabeled samples for each labeled sample
* `seed`: Seed for selecting labeled and unlabeled data

# Attributions
This thesis was done at the Centre for Mathematical Sciences at Lund University in collaboration with Peltarion AB (https://peltarion.com/).

Code from the following repos was used and modified:

* https://github.com/google-research/uda
* https://github.com/HasnainRaz/FC-DenseNet-TensorFlow
