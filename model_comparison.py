import joblib
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from simple_models import heuristic
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

sns.set()

X_test = pd.read_csv('./data/test.csv', index_col=0)
y_test = X_test.pop('Cover_Type')

scaler = joblib.load('./models/scaler.save')
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

model_rf = joblib.load('./models/model_rf.save')
model_kn = joblib.load('./models/model_kn.save')
model_nn = load_model('./models/model_nn.save')

y_pred_rf = model_rf.predict(X_test)
y_pred_kn = model_kn.predict(X_test)

nn_predictions = model_nn.predict(X_test)
y_pred_nn = nn_predictions.argmax(axis=-1)

y_pred_h = []

for index, row in X_test.iterrows():
    y_pred_h.append(heuristic(row))

predictions = pd.DataFrame({
    'y_true': y_test,
    'y_h': y_pred_h,
    'y_kn': y_pred_kn,
    'y_rf': y_pred_rf,
    'y_nn': y_pred_nn
})

accuracies = {
    'y_h': accuracy_score(predictions['y_true'], predictions['y_h']),
    'y_kn': accuracy_score(predictions['y_true'], predictions['y_kn']),
    'y_rf': accuracy_score(predictions['y_true'], predictions['y_rf']),
    'y_nn': accuracy_score(predictions['y_true'], predictions['y_nn'])
}


models = {
    'y_h': 'Heuristic',
    'y_kn': 'K nearest neighbors',
    'y_rf': 'Random forest',
    'y_nn': 'Neural network'}


def accuracy_compare_plot(accuracy_tabs, headers):
    fig = plt.figure(figsize=(10, 5))
    accuracy_val = list(accuracy_tabs.values())
    plt.bar(headers, accuracy_val)

    for i in range(len(headers)):
        plt.text(i,  accuracy_val[i]/2, f"{str(round(accuracy_val[i] * 100, 2))}%", ha='center')

    plt.title("Classification accuracy for different models")
    plt.ylabel("Accuracy")
    fig.show()


accuracy_compare_plot(accuracies, models.values())


def confusion_matrix_plot(y_true, y_compare, model_name):
    class_dict = {1: 'Spruce/Fir',
                  2: 'Lodgepole Pine',
                  3: 'Ponderosa Pine',
                  4: 'Cottonwood/Willow',
                  5: 'Aspen',
                  6: 'Douglas-fir',
                  7: 'Krummholz'}

    cm_fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(f'{model_name} - confusion matrix')
    matrix = confusion_matrix(y_true, y_compare)
    plot_confusion_matrix(matrix, class_names=class_dict, figure=cm_fig)
    cm_fig.show()


for model in models:
    confusion_matrix_plot(predictions['y_true'], predictions[model], models[model])

