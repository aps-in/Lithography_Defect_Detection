import torch
import torch.nn as nn


class CNN(nn.Module):
    '''
    The following class defines a convolutional neural network (CNN) to act as a feature extractor for the lithography Images
    '''
    def __init__(self): 
        super(CNN, self).__init__()
        self.conv = nn.Sequential( 
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
            
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
            
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
            
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2,stride=2),
                    
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2,stride=2)               
        )
    
    def forward(self, X):
        '''
        The forward function implements the forward propagation in the network

        Parameters:
        X (tensor): A batch of input images. Shape should be (batch_size, 1, h, w)
        
        Returns:
        out (tensor): A tensor of shape (batch_size, -1) representing the flattened feature maps obtained from the input image
        '''
        batch_size = X.shape[0]
        out = self.conv(X)
        return out.reshape(batch_size, -1)

class LSTM(nn.Module):
    '''
    The LSTM class helps define a Long Short Term Memory (LSTM) Network that connnects the lithography frames to one another. 
    This class also contains a fully connected network that helps in final classification of the image. 
    '''
    def __init__(self, in_size): 
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=512, num_layers=2, batch_first=True)
        
        self.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(), 
        nn.Linear(in_features=256, out_features=3), 
        )
    
    def forward(self, X):
        # Pass the input through the LSTM layer and capture the output and hidden state
        out, _ = self.lstm(X)
        
        # Take the last time step from the output to use as input for the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out

class Model(nn.Module):
    '''
    The Model class combines the CNN and LSTM classes to create the final end-to-end model for image classification.
    '''
    def __init__(self, cnn_model, lstm_model, embed_size=512): 
        super(Model, self).__init__()
        self.cnn = cnn_model
        self.lstm = lstm_model
        self.embed_size = embed_size
        
    def forward(self, X):
        batch_size, sequence_length, h, w = X.shape
        lstm_input = torch.zeros((batch_size, sequence_length, self.embed_size), device=X.device)
        
        # Iterate through each frame in the input sequence
        for frame_num in range(sequence_length):
            # Pass the current frame through the CNN to get the feature embedding
            lstm_input[:, frame_num, :] = self.cnn(X[:, frame_num, :, :].unsqueeze(1))
            
        # Pass the sequence of feature embeddings through the LSTM
        out = self.lstm(lstm_input)
        
        return out






    
   

