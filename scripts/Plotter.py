from tensorflow.keras.models import load_model

# Load the model
model = load_model('model_se_net.h5')

# Plotting the loss and accuracy curves over epochs
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy values
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('training_history.png', dpi=300) # Saving the plot to a file
    plt.show()

# Assuming history is the return value from the fit method
# replace 'history' with the variable holding your training history
history = model.fit(train_gen_combined, validation_data=validation_generator, 
                    steps_per_epoch=(len(df) + dir_train_generator.samples) // BATCH_SIZE, 
                    epochs=150) 

plot_training_history(history)

## Confusion matrix and classification report
# Getting the true labels and the predicted labels
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Generating a confusion matrix
conf_mat = confusion_matrix(true_labels, predicted_labels)

# Plotting the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=unique_classes_test,
            yticklabels=unique_classes_test)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300) # Saving the plot to a file
plt.show()

# Printing the classification report
print(classification_report(true_labels, predicted_labels, target_names=unique_classes_test))


## Visualizing predictions on individual images
# Plotting individual predictions on test images
def plot_predictions(model, generator, num_examples):
    for i in range(num_examples):
        x, y = next(generator)
        plt.imshow(x[0])
        plt.title(f"True: {np.argmax(y[0])}, Predicted: {np.argmax(model.predict(x)[0])}")
        plt.savefig(f'prediction_{i}.png', dpi=300) # Saving each plot to a separate file

        plt.show()

plot_predictions(model, test_generator, 5) # Change 5 to the number of examples you want to visualize
