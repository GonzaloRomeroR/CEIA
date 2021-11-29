from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC_AUC = {}\n".format(roc_auc))
    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize="all")
    return model, roc_auc
