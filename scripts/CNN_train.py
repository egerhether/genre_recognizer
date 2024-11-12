import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy

from CNN import CNN
from MLP_train import evaluate
from dataset import FMA_Dataset

def train_and_eval(fc_in, subset = "small", mode = "top", n_filters = [4, 12, 24, 10, 5], kernel_sizes = [5, 3, 3, 3, 3], use_batch_norm = True, dropout = 0.2, epochs = 10):
    '''
    Performs training of and evaluates the neural network genre classifier.

    Args:
      subset: string, either small, medium or large, size of the dataset used for the training
      arch: string, either mlp or cnn, architecture of the model used
      mode: string, either top or all, determines if only the top genre or all songs genres are
            used for classification training
      n_hidden: list of ints, list of numbers of neurons for hidden layers of the model
      use_batch_norm: boolean, determines whether batch normalization is used in the model
      dropout: float, size of dropout used after each layer
      epochs: int, number of training epochs

    Returns:
      model: trained model
    '''

    if use_batch_norm:
        print(f"Training CNN, batch norm and dropout of {dropout}" )
    else:
        print(f"Training CNN, no batch norm and dropout of {dropout}" )

    
    torch.manual_seed(10)

    train_data = FMA_Dataset("training", subset, "cnn", mode)

    train_loader = DataLoader(train_data, batch_size = 256, shuffle = True)
    val_loader = DataLoader(FMA_Dataset("validation", subset, "cnn", mode), batch_size = 256, shuffle = False)
    test_loader = DataLoader(FMA_Dataset("test", subset, "cnn", mode), batch_size = 256, shuffle = False)

    # Necessary for dynamic initialization, varies by subset size
    n_inputs = train_data.n_inputs
    n_classes = train_data.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and the loss
    model = CNN(n_filters, n_classes, kernel_sizes, fc_in, use_batch_norm, dropout)
    loss_module = nn.CrossEntropyLoss()
    best_val_acc = 0

    model.to(device)
    # TODO: optimize the optimizer :0
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 5e-5)

    for epoch in range(epochs):

        # Put the model into training mode
        model.train()

        for data, labels in tqdm(train_loader, desc = f"Epochs: {epoch + 1}/{epochs}"):
            
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            pred_labels = model.forward(data)
            loss = loss_module(pred_labels, labels)
            loss.backward()
            optimizer.step()

        
        val_acc = evaluate(model, val_loader)
        print(f"Validation accuracy {val_acc * 100:.2f}%")
        if val_acc > best_val_acc:
            best_model = deepcopy(model)
            best_val_acc = val_acc

    test_acc = evaluate(best_model, test_loader)
    print(f"Test accuracy {test_acc * 100:.2f}%")

    return best_model


if __name__ == "__main__":

    trained_model = train_and_eval(336, 'large', "top", [8, 24, 128, 92, 64, 16], [11, 7, 5, 3, 3, 3], True, 0.1, 150)
    torch.save(trained_model.state_dict(), "trained_models/CNN_5936.pth")
    #file = open("trained_models/MLP_5958.txt")
    #file.write()