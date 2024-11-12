import torch.nn as nn

class CNN(nn.Module):
    '''
    Class implementing a Convolutional Neural Network using PyTorch.
    '''

    def __init__(self, n_inputs, n_hidden, n_classes, kernel_sizes, use_batch_norm = False, dropout = 0.2):
        super(CNN, self).__init__()

        layers = []

        n_hidden.append(n_classes)

        input_layer = nn.Conv2d(n_inputs, n_hidden, kernel_sizes[0])
        nn.init.kaiming_uniform_(input_layer.weight)
        layers.append(input_layer)

        for i, n in enumerate(n_hidden):
            
            if i == len(n_hidden) - 1:
                break
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_hidden[i]))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layer = nn.Conv2d(n, n_hidden[i + 1], kernel_sizes[i + 1])
            nn.init.kaiming_uniform_(layer.weight)

            layers.append(layer)
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
                  
        out = x
        
        for layer in self.layers:
            out = layer(out)
            
        return out   
    
    @property
    def device(self):
        """
        Returns the device on which the model is. 
        """
        return next(self.parameters()).device