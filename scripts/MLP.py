import torch.nn as nn

class MLP(nn.Module):
    """
    Class implementing a Multi-Layer Perceptron using PyTorch.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm = True, dropout = 0.2):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
          dropout: float, defines the fraction of dropout after each later. 
                          if 0, no dropout is added.
        """

        super(MLP, self).__init__()

        layers = []
        n_hidden.append(n_classes)

        input_layer = nn.Linear(n_inputs, n_hidden[0])
        nn.init.kaiming_normal_(input_layer.weight)
        layers.append(input_layer)

        for i, n in enumerate(n_hidden):
            
            if i == len(n_hidden) - 1:
                break
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_hidden[i]))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layer = nn.Linear(n, n_hidden[i + 1])
            nn.init.kaiming_normal_(layer.weight)

            layers.append(layer)
        
        self.layers = nn.Sequential(*layers)

          

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

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