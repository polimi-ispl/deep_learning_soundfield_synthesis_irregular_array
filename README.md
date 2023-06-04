#Compensation of Driving Signals for Soundfield Synthesis through Irregular Loudspeaker Arrays Based on Convolutional Neural Network

Code repository for the paper _Compensation of Driving Signals for Soundfield Synthesis through Irregular Loudspeaker Arrays Based on Convolutional Neural Networks_
[[1]](#references).

- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Dependencies
- Python, it has been tested with version 3.6.9
- Numpy, scikit-image, scikit-learn, tqdm, matplotlib
- jax
- Tensorflow 2.4.1
- [sfs](https://sfs-python.readthedocs.io/en/0.6.2/)

### Data generation
There are two different scripts to generate the data for the circular and linear array case, namely _generate_data_circular_array.py_ and _generate_data_linear_array.py_. The parameters used for data generation (e.g. sources position, array position) are defined into _data_lib/params_circular_ and _data_lib/params_linear_, respectively.

The command-line arguments are the following
- gt_soundfield: bool, True if want to generate also data related to ground truth soundfield
- n_missing: Int, number of missing loudspeaker from full regular setup
- dataset_path: String, folder where to store dataset

### Network training
There are two different scripts to generate the data for the circular and linear array case, namely _generate_data_circular_array.py_ and _generate_data_linear_array.py_. The parameters used for data generation (e.g. sources position, array position) are defined into _data_lib/params_circular_ and _data_lib/params_linear_, respectively.
- epochs: Int, number of epochs 
- batch_size: Int, dimension of batches used for training
- log_dir: String, folder where store training logs accessible via [Tensorboard](https://www.tensorflow.org/tensorboard)
- gt_soundfield_dataset_path: String, path to numpy array where ground truth soundfield data are contained
- learning_rate: Float, learning rate used for the optimizer
- green_function: String, path to numpy array where green function between secondary sources and evaluation points is contained

### Results computation
To compute both Normalized Reproduction Error (NRE) and Structural Similarity Index (SSIM) run the code contained into _plot_results.py_ with the following arguments

- dataset_path: String, path to folder where to save results stored into arrays.
- array_type: String, type of array, i.e. linear or circular
- n_missing: Int, number of missing loudspeaker from full regular setup 

N.B. pre-trained models used to compute the results shown in [[1]](#references) can be found in folder _models_

# References
>[1] L.Comanducci, F.Antonacci, A.Sarti, Compensation of Driving Signals for Soundfield Synthesis through Irregular Loudspeaker Arrays Based on Convolutional Neural Networks [[arXiv preprint]()].
ì