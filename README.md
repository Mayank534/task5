
# Image Classifier
## Task 5

The code is made using python(model.py) and aims to train on CIFAR-10 dataset and test it. 

### Following need to be installed in your local machine to run the code:
 - numpy
 - pandas
 - argparse
 - os
 - cv2
 - pickle


### To install cv2- 
    pip install opencv-python    

### To install argparse
    pip install argparse      

### Shortcomings 
- The model is not complete, it does not run through run.sh , Although the model with not produce output it can be fixed with minor tweaking 
- The main problem I encountered was beacuse of the CIFAR-10 dataset, it was not working with any software I tried (vscode,googlecolab,jupyter nb)
- The dataset on unziping gives in the format of images and hence is difficult to use.
- Setting it up consumed most of my time.
- Does not contain nag optimizer

### Explanation of my code

#### Import
Fisrt I installed all the dependencies and then imported them 
#### Data path 
I defined the data path from my local machine to test the code. I then loaded the data path a function
### Class Neural Network
- Initialized the parameters form the input
- Made weights and biases

- Made the activation function from sratch
- Made the optimizer functions gradient descent, momentum, adam

- made forward and backward pass

### Training Model
- Trained the model
- set the loss function

### Calling main
- taking bash input from argparse and making the network using those parameters
