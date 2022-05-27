# HappyWhale

This is our repo holding the essential scripts for our assignment ! 
 - `HappyWhale1.ipynb` is a script containing our baseline architecture and the cross validation training implementation.
 - `Image recropping.ipynb` contains the preprocessing applied to the training dataset
 - `ResNet50Classifier.py` contains our main baseline model and the training loop we used to get 0.91 accuracy on the training set.
 - `SupCon.py` is our supervised contrastive learning script
 - `rembg_test.py` contains the pipeline to apply the rembg library for cropping and background removal
 - `siamese eval.ipynb` is a notebook where we train a MLP classifier over the freezed embedding network trained with SupCon.py
 - `utils.py` just contains a function to print out the images.
