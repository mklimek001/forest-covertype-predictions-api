from best_params_for_nn_preparation import create_model, find_best_params
import matplotlib.pyplot as plt
import pandas as pd

# print(find_best_params())
# {'batch_size': 100, 'epochs': 100, 'lr': 0.001}

EPOCHS = 100
BATCH_SIZE = 100
LR = 0.001


X_train = pd.read_csv('../data/train.csv', index_col=0)
y_train = X_train.pop('Cover_Type')

X_valid = pd.read_csv('../data/valid.csv', index_col=0)
y_valid = X_valid.pop('Cover_Type')


def validation_and_accuracy_graphs(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


model = create_model(LR)

hist = model.fit(X_train, y_train,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 validation_data=(X_valid, y_valid))

validation_and_accuracy_graphs(hist, EPOCHS)

model.save('../models/model_nn.save')
