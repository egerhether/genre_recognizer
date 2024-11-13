import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy


from MLP import MLP
from dataset import FMA_Dataset


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
    loss_module = nn.CrossEntropyLoss()
    loss = 0
    acc = 0
    n = 0

    with torch.no_grad():

        for data, labels in dataloader:

            data, labels = data.to(device), labels.to(device)
            pred_labels = model.forward(data)

            acc += data.shape[0] * accuracy(pred_labels, labels)
            loss += loss_module(pred_labels, labels)

            n += data.shape[0]

        acc /= n
        loss /= len(dataloader)

    return acc, loss.cpu().detach()




def train_and_eval(subset = "small", mode = "top", n_hidden = [128], use_batch_norm = True, dropout = 0.2, epochs = 10):
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
        print(f"Training MLP, batch norm and dropout of {dropout}" )
    else:
        print(f"Training MLP, no batch norm and dropout of {dropout}" )

    
    torch.manual_seed(10)

    train_data = FMA_Dataset("training", subset, "mlp", mode)

    train_loader = DataLoader(train_data, batch_size = 256, shuffle = True)
    val_loader = DataLoader(FMA_Dataset("validation", subset, "mlp", mode), batch_size = 256, shuffle = False)
    test_loader = DataLoader(FMA_Dataset("test", subset, "mlp", mode), batch_size = 256, shuffle = False)

    # Necessary for dynamic initialization, varies by subset size
    n_inputs = train_data.n_inputs
    print(n_inputs)
    n_classes = train_data.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and the loss
    model = MLP(n_inputs, n_hidden, n_classes, use_batch_norm, dropout) 
    loss_module = nn.CrossEntropyLoss()
    best_val_acc = 0

    model.to(device)
    # TODO: optimize the optimizer :0
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 5e-5)

    # for plotting
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):

        # Put the model into training mode
        model.train()
        epoch_train_loss = 0

        for data, labels in tqdm(train_loader, desc = f"Epoch: {epoch + 1}/{epochs}"):
            
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            pred_labels = model.forward(data)
            loss = loss_module(pred_labels, labels)
            epoch_train_loss += loss
            loss.backward()
            optimizer.step()

        
        epoch_train_loss = epoch_train_loss.cpu().detach() / len(train_loader)
        val_acc, val_loss = evaluate(model, val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        print(f"Validation accuracy: {val_acc * 100:.2f}%, Validation loss: {val_loss}")
        if val_acc > best_val_acc:
            best_model = deepcopy(model)
            best_val_acc = val_acc

    test_acc, test_loss = evaluate(best_model, test_loader)
    print(f"Test accuracy {test_acc * 100:.2f}%, Test loss: {test_loss}")

    plot_dict = {"test_acc": test_acc, "val_acc": val_accs, "train_loss": train_losses, "val_loss": val_losses}

    return best_model, plot_dict


if __name__ == "__main__":

    trained_model, plot_dict = train_and_eval('large', "top", [275, 160], True, 0.5, 50)
    torch.save(trained_model.state_dict(), "trained_models/MLP_6080.pth")

    plt.plot(plot_dict["train_loss"])
    plt.plot(plot_dict["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.show()
    #file = open("trained_models/MLP_5958.txt")
    #file.write()