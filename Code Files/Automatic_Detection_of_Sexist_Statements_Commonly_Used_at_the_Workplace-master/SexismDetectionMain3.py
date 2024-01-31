from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, TimeDistributed, Flatten, Multiply, Concatenate#, GlobalAveragePooling1D, regularizers
from keras.layers import Embedding
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import load_model, Model, Sequential
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler
# import keras.backend as K

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


from SD_utils import *
from nmt_utils import *

import matplotlib.pyplot as plt

import os
import io
import pprint as pp
import gc
import numpy as np
import re

np.random.seed(0)
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz-2.38/bin'

sexist_dataset_fn = 'data/shuffled_data.csv'
embedding_fn = 'data/vectors.txt'

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(embedding_fn)

X, y = read_csv(sexist_dataset_fn)

############ Fine Tuning ###############

general_hate_speech_dataset1 = 'data/cleaned_general_hate_speech_dataset.csv'
# general_hate_speech_dataset1 = 'data/MHS dropped ambiguous and neutral.csv'
X_general1, y_general1 = read_csv(general_hate_speech_dataset1)

# Split each general hate speech dataset into training, validation, and testing sets
X_general1_train, X_general1_temp, y_general1_train, y_general1_temp = train_test_split(X_general1, y_general1, test_size=0.2, random_state=np.random)
X_general1_val, X_general1_test, y_general1_val, y_general1_test = train_test_split(X_general1_temp, y_general1_temp, test_size=0.5, random_state=np.random)

m = len(y)
maxLen = len(max(X, key=len).split()) + 1

def lr_schedule(epoch):
    initial_learning_rate = 0.001  # Set your initial learning rate
    drop = 0.5  # Set the factor by which the learning rate should drop
    epochs_drop = 10  # Set the number of epochs after which to reduce the learning rate

    if epoch % epochs_drop == 0:
        new_lr = initial_learning_rate * (drop ** (epoch // epochs_drop))
        return new_lr
    return initial_learning_rate


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    # sentence = re.sub(r'[^\w\s]', '', X_s[i].lower())

    words = sentence.lower().split()
    avg = np.zeros(50)
    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
    avg = avg / len(words)

    return avg


def sentences_to_indices(X_s, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X_s.shape[0]  # number of training examples

    X_indices = np.zeros((m, max_len))  # If there is more than one dimension use ()

    for i in range(m):  # loop over training examples
        # sentence = re.sub(r'[^\w\s]', '', X_s[i].lower())
        sentence_words = X_s[i].lower().split()

        j = 0
        for w in sentence_words:
            if j < max_len and w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            j = j + 1

    return X_indices



def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 2  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)
    print(emb_dim)

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

########### Model 4 with GRU instead of LSTM #############
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, n_features=maxLen,
            isRandom=False, isAttention=True):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    # gru_layer = Bidirectional(GRU(lay1_num, return_sequences=True), input_shape=input_shape)
    # X = gru_layer(embeddings)

    X = Bidirectional(GRU(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
    X = Dropout(0.5)(X)
    if isAttention:
        attention = Dense(1, activation='tanh')(X)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(lay1_num * 2)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([X, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

        X = Dropout(0.5)(sent_representation)
    else:
        X = Bidirectional(GRU(lay2_num, return_sequences=False), input_shape=input_shape)(X)
        X = Dropout(0.5)(X)

    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)

    # SVM Classifier
    svm_output = Dense(2, activation='linear', name='svm_output')(X)  # Linear activation for SVM

    svm_model = Model(inputs=sentence_indices, outputs=svm_output)

    model = Model(inputs=sentence_indices, outputs=X)

    return model, svm_model


###### To predict if a sentence is sexist or not ###########
def predict_sexism_v4(model, speech, word_to_vec_map, word_to_index, maxLen):
    indices = sentences_to_indices(speech, word_to_index, max_len=maxLen)
    prediction = model.predict(indices)
    return int(np.argmax(prediction))

# Function to train the model
def train_model(X_train_indices, y_train_oh, X_val_indices, y_val_oh, maxLen, word_to_vec_map, word_to_index):
    model, _ = modelV4((maxLen,), word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, n_features=maxLen, isRandom=False, isAttention=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('weights/modelWeights_updated.h5', save_best_only=True, save_weights_only=True)

    hist = model.fit(X_train_indices, y_train_oh, validation_data=(X_val_indices, y_val_oh), epochs=100, batch_size=8, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])

    # Print loss and accuracy graphs
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig('accAndLossPlots/acc_and_loss_plot_final.png')
    plt.show()

    return model

def evaluate_model(model, X_test, y_test, word_to_vec_map, word_to_index, maxLen):
    v4_tp, v4_tn, v4_fp, v4_fn = 0, 0, 0, 0

    # Convert sentences to indices for the test set
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)

    # Predictions using the current model
    predV2 = model.predict(X_test_indices)

    # Loop over each example in the test set
    for k in range(len(X_test)):
        num = np.argmax(predV2[k])
        if num != y_test[k]:
            if X_test[k] in wrongs_dict:
                wrongs_dict[X_test[k]] += 1
            else:
                wrongs_dict[X_test[k]] = 1
            if int(num) == 1:
                v4_fp += 1
            elif int(num) == 0:
                v4_fn += 1
        else:
            if int(num) == 1:
                v4_tp += 1
            elif int(num) == 0:
                v4_tn += 1

    # Update the counters
    tp_sum_v[4] += v4_tp
    tn_sum_v[4] += v4_tn
    fp_sum_v[4] += v4_fp
    fn_sum_v[4] += v4_fn

    # Calculate metrics
    precision = v4_tp / (v4_tp + v4_fp) if v4_tp + v4_fp != 0 else 0
    recall = v4_tp / (v4_tp + v4_fn) if v4_tp + v4_fn != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    accuracy = accuracy_score(y_test, np.argmax(predV2, axis=1))

    # Print metrics
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)
    print("Accuracy: ", accuracy)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, np.argmax(predV2, axis=1))

    # Calculate percentages
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Display the confusion matrix as percentages
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_percent, cmap='Blues', interpolation='nearest')

    # Add colorbar
    plt.colorbar()

    # Set labels
    classes = ["Class 0", "Class 1"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)  # Removed rotation argument
    plt.yticks(tick_marks, classes)

    # Add annotations with adjusted text color
    for i in range(conf_matrix_percent.shape[0]):
        for j in range(conf_matrix_percent.shape[1]):
            text_color = 'black' if conf_matrix_percent[i, j] < 0.5 else 'white'  
            plt.text(j, i, f'{conf_matrix_percent[i, j]*100:.2f}%', ha='center', va='center', color=text_color)

    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add title
    plt.title('Confusion Matrix (Percentages)')

    # Save the figure
    plt.savefig('accAndLossPlots/confusionMatrixFinal.png')
    plt.show()

    # Plot the model architecture
    plot_model(model, to_file='modelPlot/model_v4.png')

def incremental_train_model(model, X_new_indices, y_new_oh, maxLen, word_to_vec_map, word_to_index):
    # Train the model with new data
    model.fit(X_new_indices, y_new_oh, epochs=5, batch_size=8, shuffle=True)
    
    # Save the updated model weights
    model.save_weights('weights/modelWeights_updated_incremental.h5')

    return model

tests_dict, wrongs_dict = {}, {}
num_it = 1
n_models = 4
acc_sum_v = [0] * (n_models + 1)
fp_sum_v, fn_sum_v = [0] * (n_models + 1), [0] * (n_models + 1)
tp_sum_v, tn_sum_v = [0] * (n_models + 1), [0] * (n_models + 1)
num_0s_sum, num_1s_sum = 0, 0

runModels = [None,
             False,  # run ModelV1?
             False,  # run ModelV2?
             False,  # run ModelV3?
             True,] # run ModelV4?

gb_accs = [0, 0, 0]

if __name__ == "__main__":
    for it in range(num_it):
        # print("\nIteration #" + str(it + 1))

        X, y = read_csv(sexist_dataset_fn)

        # Split the dataset into training (80%) and combined validation/testing (20%) sets
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=np.random)

        # Split the validation/testing set into validation (50%) and testing (50%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=np.random)

        maxLen = len(max(X, key=len).split()) + 1
        y_oh_train = convert_to_one_hot(y_train, C=2)
        y_oh_test = convert_to_one_hot(y_test, C=2)
        y_oh_val = convert_to_one_hot(y_val, C = 2)

        num_1s = sum(y_test)
        num_0s = y_test.shape[0] - num_1s
        num_0s_sum += num_0s
        num_1s_sum += num_1s

        # V4 Model
        if runModels[4]: 
            # Convert sentences to indices for both training and validation sets
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            X_val_indices = sentences_to_indices(X_val, word_to_index, maxLen)

            # Convert labels to one-hot encoding for both training and validation sets
            y_train_oh = convert_to_one_hot(y_train, C=2)
            y_val_oh = convert_to_one_hot(y_val, C=2)

            # Train the initial model or load weights based on the flag
            train_model_flag = False 
            incremental_flag = False
            user_predictions_flag = False

            if train_model_flag:
                model_v4 = train_model(X_train_indices, y_train_oh, X_val_indices, y_val_oh, maxLen, word_to_vec_map, word_to_index)
            else:
                # Load weights if the flag is set to False
                model_v4, _ = modelV4((maxLen,), word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, isRandom=False, isAttention=True)
                # model_v4.load_weights('weights/GRU/0.769.h5')
                # model_v4.load_weights('weights/gru, data aug, epochs 100, 64 batch/m4_weights_final_data.h5')
                model_v4.load_weights('weights/modelWeights.h5')

            user_inputs = []
            user_corrections = []

            while True:
                user_input = input("Enter a statement (or type 'done' to finish input): ")
                if user_input.lower() == 'done':
                    break

                if user_predictions_flag:
                    user_correction = input("Enter the correct label ('sexist' or 'not sexist') for the statement: ")

                    # Convert user input to numerical label
                    user_correction_numeric = 1 if user_correction.lower() == 'sexist' else 0

                    user_corrections.append(user_correction_numeric)

                user_inputs.append(user_input)

            # Predict using the current model for all collected inputs
            current_predictions = [predict_sexism_v4(model_v4, np.array([input_text]), word_to_vec_map, word_to_index, maxLen) for input_text in user_inputs]

            # Display model predictions
            print("Model predictions:")
            for input_text, prediction in zip(user_inputs, current_predictions):
                # Convert numerical prediction to 'sexist' or 'not sexist'
                prediction_label = 'sexist' if prediction == 1 else 'not sexist'
                print(f"Input: {input_text}, Prediction: {prediction_label}")

            if user_predictions_flag:
                # Check user corrections and update the model if needed
                corrections_needed = [user_correction != current_prediction for user_correction, current_prediction in zip(user_corrections, current_predictions)]

                # Update training data and retrain the model if corrections are needed
                if any(corrections_needed):
                    if incremental_flag:  # Use incremental learning instead of retraining with the entire dataset
                        # Convert new statements to indices
                        X_new_indices = sentences_to_indices(user_inputs, word_to_index, maxLen)

                        # Convert new labels to one-hot encoding
                        y_new_oh = convert_to_one_hot(user_corrections, C=2)
                        model_v4 = incremental_train_model(model_v4, X_new_indices, y_new_oh, maxLen, word_to_vec_map, word_to_index)

                    else:  # Retrain the model with all collected corrections
                        # Print new training samples before appending to X_train and y_train
                        print("New Training Samples:")
                        for user_input, user_correction in zip(user_inputs, user_corrections):
                            print(f"Input: {user_input}, Label: {user_correction}")

                        # Append statements and labels to the training data
                        user_inputs_array = np.array(user_inputs, dtype='<U315')
                        X_train = np.concatenate((X_train, user_inputs_array))
                        y_train = np.append(y_train, user_corrections)

                        # Convert statements to indices
                        X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

                        # Convert labels to one-hot encoding
                        y_train_oh = convert_to_one_hot(y_train, C=2)
                        model_v4 = train_model(X_train_indices, y_train_oh, X_val_indices, y_val_oh, maxLen, word_to_vec_map, word_to_index)

                    # Identify new training samples and append them to the original dataset file
                    new_training_samples = list(zip(user_inputs, user_corrections))
                    original_dataset = pd.read_csv('data/shuffled_data.csv')
                    new_training_samples_df = pd.DataFrame(new_training_samples, columns=original_dataset.columns)

                    # Load the original dataset
                    original_dataset = pd.read_csv('data/shuffled_data.csv')

                    # Append new training samples to the original dataset
                    original_dataset = pd.concat([original_dataset, new_training_samples_df], ignore_index=True)

                    # Write the updated dataset back to the file
                    original_dataset.to_csv('data/shuffled_data.csv', index=False)

                    evaluate_model(model_v4, X_test, y_test, word_to_vec_map, word_to_index, maxLen)
                    print("Model retrained!")
                else:
                    print("No corrections needed!")