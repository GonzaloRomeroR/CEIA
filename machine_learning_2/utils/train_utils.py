from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from functools import wraps
from time import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from utils.plot_utils import plot_roc_curve
import torch
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def evaluate_model_mlflow(train_model, X_train, y_train, X_test, y_test, params, models, model_name, experiment_id):
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):
        
        model = Model(train_model, {})
        perform_grid_search(model, X_train, y_train, params)

        params = model.best_params  
        for param in params:
            mlflow.log_param(param, params[param])

        model = Model(train_model, params)
        evaluate_model(model, X_train, y_train, X_test, y_test)
        models[model_name] = model

        mlflow.log_metric("auc", model.roc_auc)
        mlflow.log_metric("precision", model.report["weighted avg"]["precision"])
        mlflow.log_metric("recall", model.report["weighted avg"]["recall"])
        mlflow.log_metric("f1-score", model.report["weighted avg"]["f1-score"])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, model_name, registered_model_name="model")
        else:
            mlflow.sklearn.log_model(model, model_name)
        
        return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)
    model.classification_report(X_test, y_test)
    print("Training time: {} s".format(model.train_time))
    model.plot_curve(X_test, y_test)


def perform_grid_search(model, X_train, y_train, params):
    model.grid_search(X_train, y_train, params)
    print("Grid search time: {} s".format(model.grid_time))


class Model:
    def __init__(self, model_class, params):
        self.model = model_class(**params)
        self.train_time = None

    def train(self, X, y):
        start = timer()
        self.model.fit(X, y)
        end = timer()
        self.train_time = end - start

    def grid_search(self, X, y, params):
        start = timer()
        grid = GridSearchCV(self.model, params)
        grid.fit(X, y)
        end = timer()
        self.best_params = grid.best_params_
        print(self.best_params)
        self.grid_time = end - start

    def predict(self, X):
        return self.model.predict(X)

    def classification_report(self, X, y):
        y_pred = self.predict(X)
        roc_auc = roc_auc_score(y, y_pred)
        self.roc_auc = roc_auc
        print("ROC_AUC = {}\n".format(roc_auc))
        print(classification_report(y, y_pred, digits=5))
        self.report = classification_report(y, y_pred, digits=5, output_dict=True)
        return self.report

    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.grid(None)

    def plot_curve(self, X, y):
        y_pred = self.predict(X)
        plot_roc_curve(y, y_pred)


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC_AUC = {}\n".format(roc_auc))
    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize="all")
    plt.grid(None)
    return model, roc_auc


def fit_model(model, criterion, optimizer, dataloaders, train_len, test_len, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        model.cuda()
    losses_train = []
    losses_val = []
    running_loss = 0.0
    start = timer()
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
    end = timer()
    train_time = start - end
    return losses_train, losses_val, train_time
