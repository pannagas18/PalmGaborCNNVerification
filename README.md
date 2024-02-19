# PALMGABORCNNVERIFICATION

The project was inspired by [PalmNet](https://ieeexplore.ieee.org/document/8691498). 

The feature extraction is modified to a CNN network for a much better representation of the image which is then trained using a Siamese Network with TripletLoss function. The network and the loss functions are modified to accomodate the complex numbered vectors that are obtained by applying the Gabor Filters.

The model can be infered from [inference.py](https://github.com/pannagas18/PalmGaborCNNVerification/blob/main/inference.py).
