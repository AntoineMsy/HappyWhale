# HappyWhale

This is our repo holding the essential scripts for our assignment ! 
 - `Image recropping.ipynb` contains the preprocessing applied to the training dataset
 - `ResNet50Classifier.py` contains our main baseline model with 0.91 accuracy on the training set
 - `SupCon.py` is our supervised contrastive learning script
 - `rembg_test.py` contains the pipeline to apply the rembg library for cropping and background removal
 - `siamese eval.ipynb` is a notebook where we train a MLP classifier over the freezed embedding network trained with SupCon.py
 - `utils.py` just contains a function to print out the images

PS : the .pt files containing the models' statedicts were too big to be uploaded on github, as well as the data ;)
