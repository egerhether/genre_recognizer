import torch.nn as nn
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy


from MLP import MLP
from CNN import CNN
from dataset import FMA_Dataset
from genre_utils import id_to_genre, genres


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """
    y_pred = torch.argmax(predictions, axis = 1)

    accuracy = 0
    for pred, label in zip(y_pred, targets):
        if pred == label:
            accuracy += 1

    accuracy /= len(targets)

    return accuracy

def evaluate(model, dataloader):

    model.eval()
    device = model.device
    acc = 0
    n = 0

    with torch.no_grad():

        for data, labels in dataloader:

            data, labels = data.to(device), labels.to(device)
            pred_labels = model.forward(data)
            acc += data.shape[0] * accuracy(pred_labels, labels)
            n += data.shape[0]

        acc /= n

    return acc




def train_and_eval(subset = "small", arch = "mlp", mode = "top", n_hidden = [128], use_batch_norm = True, dropout = 0.2, epochs = 10):
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
        print(f"Training using model {arch}, batch norm and dropout of {dropout}" )
    else:
        print(f"Training using model {arch}, no batch norm and dropout of {dropout}" )

    
    torch.manual_seed(10)

    train_data = FMA_Dataset("training", subset, mode)

    train_loader = DataLoader(train_data, batch_size = 256, shuffle = True)
    val_loader = DataLoader(FMA_Dataset("validation", subset, mode), batch_size = 256, shuffle = False)
    test_loader = DataLoader(FMA_Dataset("test", subset, mode), batch_size = 256, shuffle = False)

    raise Exception()

    # Necessary for dynamic initialization, varies by subset size
    n_inputs = train_data.n_inputs
    n_classes = train_data.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and the loss
    if arch in ["mlp", "cnn"]:
        model = MLP(n_inputs, n_hidden, n_classes, use_batch_norm, dropout) if arch == "mlp" else CNN()
    else:
        raise Exception("Provide a correct training architecture - \"mlp\" or \"cnn\" ")
    
    best_val_acc = 0

    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    # TODO: optimize the optimizer :0
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 1e-4)

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

    trained_model = train_and_eval('large', "mlp", "top", [164], True, 0.3, 50)
    torch.save(trained_model.state_dict(), "trained_models/MLP_5958.pth")
    #file = open("trained_models/MLP_5958.txt")
    #file.write()