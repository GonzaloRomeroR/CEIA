from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from functools import wraps
from time import time
import matplotlib.pyplot as plt
import torch


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC_AUC = {}\n".format(roc_auc))
    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize="all")
    plt.grid(None)
    return model, roc_auc


def fit(model, criterion, optimizer, dataloaders, train_len, test_len, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        model.cuda()
    losses_train = []
    losses_val = []
    running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        correct_val = 0
        model = model.train()
        for i, data in enumerate(dataloaders["train"]):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            preds = torch.round(torch.sigmoid(outputs))
            optimizer.step()
            running_loss += loss.item()
            correct += (preds == labels.unsqueeze(1)).float().sum()
        model = model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, data in enumerate(dataloaders["val"]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                correct_val += (preds == labels.unsqueeze(1)).float().sum()

        print(
            "[%d] training loss: %.5f, validation loss: %.5f"
            % (
                epoch + 1,
                running_loss / len(dataloaders["train"]),
                valid_loss / len(dataloaders["val"]),
            ),
            end="",
        )
        print(
            ", train accuracy {}, val accuracy {}".format(
                100 * correct / train_len, 100 * correct_val / test_len
            )
        )
        losses_train.append(running_loss / len(dataloaders["train"]))
        losses_val.append(valid_loss / len(dataloaders["val"]))
    return losses_train, losses_val


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_t = te - ts
        return result, time_t

    return wrap
