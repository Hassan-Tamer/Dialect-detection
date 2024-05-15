import tensorflow as tf
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


NUM_CLASSES = 5
EPOCHS = 10

def loadArrays(file1,file2):
    with open(file1, 'rb') as f:
        loaded_array = pickle.load(f)

    X = loaded_array

    with open(file2, 'rb') as f:
        loaded_array = pickle.load(f)

    y = loaded_array

    return X,y


if __name__ == '__main__':
    X,y = loadArrays('X.pkl','y.pkl')

    print("len(X):", len(X))
    print("len(y):", len(y))

    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("len(X_train):", len(X_train))
    print("len(X_test):", len(X_test))
    print("len(y_train):", len(y_train))
    print("len(y_test):", len(y_test))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(124,1024), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ], name='yamnet_model')


    # Compile the model
    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=METRICS)


    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=EPOCHS//3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data = (X_test, y_test), batch_size=32,callbacks=[early_stopping])

    model.save('yamnet_modelV2.h5')
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    evaluation = model.evaluate(X_test, y_test)
    print(f'Val Accuracy : {evaluation[1] * 100:.2f}%')

    # Predict the values from the testing dataset
    Y_pred = model.predict(X)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred,axis = 1)
    # Convert testing observations to one hot vectors
    Y_true = np.argmax(y,axis = 1)
    # compute the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g')

    print(classification_report(Y_true, Y_pred_classes))
