import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from potato import test_generator, unique_classes_train  # assuming your main script is named main_script.py

import pickle

def load_history(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_training_history(history):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1) 
    plt.plot(history['accuracy'])  # use history['accuracy'], not history.history['accuracy']
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])  # use history['loss'], not history.history['loss']
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig('training_history.png')

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.savefig("confusion_matrix.png")


def evaluate_model(model, test_generator, classes):
    # Predict class indices
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)

    # Get true class indices
    y_true = test_generator.classes

    # Calculate confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes)

    # Classification report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=classes))


if __name__ == '__main__':
    history = load_history('history.pkl')
    plot_training_history(history)
    model = load_model('model_se_net.h5')  # Load the trained model
    evaluate_model(model, test_generator, unique_classes_train)