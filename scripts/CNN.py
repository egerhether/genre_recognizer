import torch.nn as nn

class CNN(nn.Module):
    '''
    Class implementing a Convolutional Neural Network using PyTorch.
    '''

    def __init__(self, n_filters, n_classes, kernel_sizes, fc_in, use_batch_norm = False, dropout = 0.2):
        super(CNN, self).__init__()

        layers = []

        input_layer = nn.Conv1d(1, n_filters[0], kernel_sizes[0])
        nn.init.kaiming_uniform_(input_layer.weight)
        layers.append(input_layer)

        for i, n in enumerate(n_filters):
            
            if i == len(n_filters) - 1:
                break
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_filters[i]))

            layers.append(nn.LeakyReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layer = nn.Conv1d(n, n_filters[i + 1], kernel_sizes[i + 1])
            nn.init.kaiming_uniform_(layer.weight)

            layers.append(layer)

            if i % 2 == 0:
                layers.append(nn.MaxPool1d(kernel_size = 3, stride = 2))

        
        if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_filters[-1]))
        
        layers.append(nn.LeakyReLU())

        output_layer = nn.Linear(fc_in, n_classes)
        nn.init.kaiming_normal_(output_layer.weight)

        layers.append(output_layer)
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
                  
        out = x
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                out = out.view(out.size(0), -1)

            out = layer(out)
            
        return out   
    
    @property
    def device(self):
        """
        Returns the device on which the model is. 
        """
        return next(self.parameters()).device