# HappyWhale

This is our repo holding the essential scripts for our assignment ! 
 - Image recropping.ipynb contains the preprocessing with the rembg library
 - ResNet50Classifier.py contains our main baseline model with 0.91 accuracy on the training set
 - SupCon is our supervised contrastive learning script
 - siamese eval.ipynb is a notebook where we train a MLP classifier over the freezed embedding network trained with SupCon.py
 - utils.py just contains a function to print out the images
