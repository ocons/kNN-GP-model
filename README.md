# kNN-GP-model

Combined k-NN and Gaussian process regressors to model the function between a multidimensional nonlinear optical signal and a reference angle.

You obtain a 32-dimensional dataset from a 32x1-pixel array inside an optical sensor. The dataset at hand has been measured running on one of our high-precision test benches for precise angle measurement. The last column (33) contains the reference angle while the columns 1 to 32 represent each pixel stream. The task is to find an algorithm which models the mathematical function between the 32x1-pixel images and the precise reference angle.

As the original dataset is 100 Mb, here we upload only a small part of the dataset. The original dataset can be provided on request.

SOLUTION

Train a K-NN regression model using 1000 images (approximately one every 0.36 degrees) with five nearest neighbours. The number of nearest neighbours can be of course easily optimised, but it is not done here. The K-NN model gives good prediction on test data, but not as good as desired. The following step is the training of two GP regression models, one (GP0) using 1000 images in range [0 degrees ,180 degrees] and another (GP1) using 1000 images in range [180 degrees, 360 degrees]. Why two? Because one does not perform well on the entire dataset.

In conclusion, to predict the angle for a new image, the K-NN model is used as a binary classifier, i.e. if the predicted angle belongs to range [0 degrees ,180 degrees] then the image belongs to class 0, otherwise it belongs to class 1. When the image belongs to class 0 or class 1, we compute its predictive distribution using GP0 or GP1, respectively. The plot represents a test on 3600 images. The average error is also calculated on 3600 images and it is around 0.22 degree. GPs provide also the uncertainty on predicted angles, which can be used to accept or discard the measure.
