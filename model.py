#importing important libraries
import numpy as np                    
import pandas as pd                   
import argparse                       
import os                           
import cv2                          
import pickle                         

#defining path 
train_path = '/home/mayank/Desktop/train'
test_path = '/home/mayank/Desktop/test'
trainLabels_path = '/home/home/Desktop/trainLabels.csv'

train = []
test = []

def load_Cifar10(train_path,test_path):
    for root,_,files in os.walk(train_path):
        for file in files:
            if file.endswith('.png'):
                #loading image using cv2
                image_path = os.path.join(root,file)
                image = cv2.imread(image_path)
                
                #converting image to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                #converting it to numpy array
                image_array = np.array(image)
                
                #adding it to training list 'train'
                train.append(image_array)

    #writing code for loading testing dataset
    for root,_,files in os.walk(test_path):
        for file in files:
            if file.endswith('.png'):
                
                image_path = os.path.join(root,file)
                image = cv2.imread(image_path)
                
                #converting image to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                #converting it to numpy array
                image_array = np.array(image)
                
                #adding it to training list 'train'
                test.append(image_array)

    train = np.array(train)
    test = np.array(test)
                
    #reading trainLabels.csv
    trainLabels = pd.read_csv(trainLabels_path)
    trainLabels = trainLabels.to_numpy()

    

# Nueral Network
class FF_NueralNetwork:
    def init(self, input_size, sizes,num_hidden, output_size,activation,momentum):
        #declaringn important variables
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = sizes
        self.num_hidden = num_hidden
        self.activation = activation
        self.momentum = momentum
    
        self.weights = []
        self.biases = []
        self.initialise_parameters()                               #I will use this function to initialise weights and biases for every layer in the network
        
    def initialise_parameters(self):
        layer_size = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(1,len(layer_size)):
            
            prev_size = layer_size[i-1]
            curr_size = layer_size[i]
            
            #initialising weight and bias for i th layer
            weight = np.random.randn(prev_size,curr_size)*0.01
            bias = np.zeroes(1,curr_size)*0.01
            
            self.weights.append(weight)
            self.biases.append(bias)
            
    #let us define activation functions - tanh and sigmoid

    #sigmoid function
    def sigmoid(self,x):                                               
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):                                 
        return self.sigmoid(x)*(1-self.sigmoid(x))

    #tanh function
    def tanh(self,x):                                                     
        return np.tanh(x)                                             

    def tanh_derivative(self,x):                                           
        return 1-np.tanh(x)**2
    
    #optimizer function
    
    #gradient descent 
    def gradient_descent(self,lr):
        for i in range(len(self.hidden_sizes)):
            self.weights[i] -= lr*self.grad_weights[i]
            self.biases[i] -= lr*self.grad_biases[i]
            
    #momentum gradient 
    def momentum_optm(self,lr):
        for i in range(len(self.hidden_sizes)):
            self.velocities[i] = lr*self.grad_weights[i] + self.momentum*self.velocities[i]
            self.weights[i] -= self.velocities[i]
            self.biases[i] -= lr*self.grad_biases[i]
    
            
    #Adaptive momentum optimizer or 'adam'
    def adam_optm(self,lr):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        for i in range(len(self.hidden_sizes)):
            self.t[i] += 1
            self.velocities[i] = beta1*self.velocities + (1-beta1)*self.grad_weights[i]
            self.velocities_corrected[i] = self.velocities[i]/(1-beta1**self.t[i])
            self.squared_velocities[i] = beta2*self.squared_velocities[i] + (1-beta2)*np.square(self.grad_weights[i])
            self.squared_velocities_corrected[i] = self.squared_velocities[i]/(1-beta2**self.t[i])
            self.weights[i] -= lr*self.velocities_corrected[i]/(np.sqrt(self.squared_velocities_corrected[i]) + epsilon)
            self.biases[i] -= lr*self.grad_biases[i]
            
    #defining forward mode of my network
    def forward(self,x):
        activations = [x]
        for i in range(len(self.weights)):
            weight = self.weight[i]
            bias = self.bias[i]
            activation = activations[-1]
            
            #Linear Transformation
            Z = np.dot(activation,weight) + bias
            
            #Activation Function
            if self.activation == "sigmoid" :
                A = self.sigmoid(Z)
            elif self.activation == "tanh" :
                A = self.tanh(Z)
            
            activations.append(A)
            
        return activations[-1]
    
    def backward(self,x,y,lr,optm):
        m = x.shape[0]
        grad_weights = [np.zeros_like(weight) for weight in self.weights]
        grad_biases = [np.zeros_like(bias) for bias in self.biases]
        
        #Compute gradients
        activations = [x]
        for i in range(len(self.weights)):
            weight = self.weight[i]
            bias = self.bias[i]
            activation = activations[-1]
            
            #Linear Transformation
            Z = np.dot(activation,weight) + bias
            
            #Activation Function
            if self.activation == "sigmoid" :
                dA = self.sigmoid_derivative(Z)
            elif self.activation == "tanh" :
                dA = self.tanh_derivative(Z)
                
            dZ = dA
            dW = np.dot(activation.T, dZ)/m
            db = np.sum(dZ, axis=0, keepdims = True)
            dA_prev = np.dot(dZ, weight.T)
            
            grad_weights[i] = dW
            grad_biases[i] = db
            activations.append(dA_prev)
            
        #updating parameters using optimizer algorithms
        if optm == "gd":
            self.gradient_descent(lr)
        elif optm == "momentum":
            self.momentum_optm(lr)
        elif optm == "adam_optm":
            self.adam_optm(lr)
            
    def train(self,x,y,lr,num_epochs, batch_size, anneal, save_dir, expt_dir, optm,loss):
        #declaring useful variables
        num_examples = x.shape[0]
        num_batches = num_examples // batch_size
        
        #starting training
        for epoch in range(num_epochs):
            #shuffle the data for more accuracy
            indices = np.random.permutation(num_examples)
            x = x[indices]
            y = y[indices]
            
            for batch in range(num_batches):
                start = batch*batch_size
                end = (batch+1)*batch_size
                batch_x = x[start:end]
                batch_y = y[start:end]
                
                #forward pass
                output = self.forward(batch_x)
                
                #backward pass
                self.backward(batch_x,batch_y,lr,optm)
                
                #calculating loss and error rate
                if loss == "sq":                                   # squared mean error                     
                    output = np.clip(output, epsilon, 1-epsilon)
                    batch_y = np.eye(10)[batch_y]   
                    error = np.mean(np.square(output-batch_y))
                elif loss == "ce":                                 #cross entropy loss error
                    epsilon =1e-15
                    output = np.clip(output, epsilon, 1-epsilon)
                    batch_y = np.eye(10)[batch_y]                  #converting batch_y to one-hot encoded matrix
                    error = -np.sum(batch_y*np.log(output))
                
                
                #print progress
                if(batch+1)%100==0:
                    print("Epoch {}, Step {}, Loss:{:4f}, Error: {:0.2f}%, lr: {}".format(epoch,batch+1,error,error*100,lr))
            
            #Annealing
            if anneal and epoch>0:
                if error<prev_error:
                    lr /= 2
                    print("Learning rate annealed to :", lr)
                prev_error = error
                
    def predict(self, x):
        return np.argmax(self.forward(x),axis=1)
    
    def save_model(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_file = os.path.join(save_dir, "model.pkl")
        with open(model_file,"wb") as f:
            pickle.dump(self,f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default = 0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default = 0.5, help="Momentum for momentum-based algorithms")
    parser.add_argument("--num_hidden", type=int, default = 3, help="Number of hidden layers")
    parser.add_argument("--sizes", type=str, default ="100,100,100", help="Sizes of each hidden layers")
    parser.add_argument("--activation", type=str, default = 'sigmoid', help="Activation Function(sigmoid/tanh)")
    parser.add_argument("--loss", type=str, default = "sq", help="Loss Function(sq/ce)")
    parser.add_argument("--opt", type=str, default = "adam", help="Optimizer function(gd/momentum/adam)")
    parser.add_argument("--batch_size", type=int, default = 20, help="Batch Size")
    parser.add_argument("--anneal", action="store true", help="Enable learning rate annealing ")
    parser.add_argument("--save_dir", type=str, default ="/pa1", help="Directory to save the model")
    parser.add_argument("--expt_dir", type=str, default = "/pa1/exp1", help="Directory to save the log files")
    parser.add_argument("--train", type=str, default = "train.csv", help="path to training dataset")
    parser.add_argument("--test", type=str, default = "test.csv", help="path to validating dataset")
    args = parser.parse_args()
    
    
    train_x, train_y, test_x, test_y = load_Cifar10(args.train,args.test)
    
    #Normaize the input data
    train_x = train_x/255.0
    test_x = test_x/255.0
    
    #convert labels to one hot encoding
    num_classes=10
    train_y = np.eye(num_classes)
    test_y = np.eye(num_classes)
    
    #Parse sizes of each hidden layer
    hidden_sizes = list(map(int, args.split(",")))
    
    #Create the nueral network
    input_size = train_x.shape[1]
    output_size = num_classes
    activation = args.activation 
    momentum = args.momentum
    nn = FF_NueralNetwork(input_size,hidden_sizes, output_size, activation, momentum)
    
    #train the nueral network
    lr = args.lr
    num_epochs = 1
    batch_size = args.batch_size
    anneal = args.anneal
    save_dir = args.save_dir
    expt_dir = args.expt_dir
    optimizer = args.opt
    loss = args.loss
    
    nn.train(train_x, train_y, lr, num_epochs, batch_size, anneal, save_dir, optimizer,loss)
    
    #save the train model
    nn.save_model(save_dir)
    

    predictions = nn.predict(test_x)
    accuracy = np.mean(predictions == np.argmax(test_y,axis=1))
    print("Test Accuracy:",accuracy)