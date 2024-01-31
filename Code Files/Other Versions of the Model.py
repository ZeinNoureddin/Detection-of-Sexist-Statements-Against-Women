########### Model 4 without SVM (Bidirectional LSTM with Attention) #############################
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, n_features = maxLen,
            isRandom=False, isAttention=True):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)  # embedding_layer(sentence_indices)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
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
        X = Bidirectional(LSTM(lay2_num, return_sequences=False), input_shape=input_shape)(X)
        X = Dropout(0.5)(X)
        pass

    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model


########### Model 4 with SVM (Bidirectional LSTM with Attention) #############################
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, n_features=maxLen, isRandom=False, isAttention=True):
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)  # embedding_layer(sentence_indices)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    # Define an intermediate layer to extract its output
    X = Bidirectional(LSTM(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
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
        X = Bidirectional(LSTM(lay2_num, return_sequences=False), input_shape=input_shape)(X)
        X = Dropout(0.5)(X)


    # SVM Classifier
    svm_output = Dense(2, activation='linear', name='svm_output')(X)  # Linear activation for SVM
    
    # Softmax Layer
    softmax_output = Dense(2, activation='softmax', name='softmax_output')(X)
    
    model = Model(inputs=sentence_indices, outputs=softmax_output)

    svm_model = Model(inputs=sentence_indices, outputs=svm_output)

    return model, svm_model


########### Model 4 with added CNN layer #############
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128, n_features=maxLen,
                     isRandom=False, isAttention=True, num_filters=64, kernel_size=3):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50, input_length=maxLen)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    # Add the CNN layer
    X = Conv1D(num_filters, kernel_size, activation='relu')(embeddings)
    X = MaxPooling1D(pool_size=2)(X)
    X = GlobalMaxPooling1D()(X)
    
    # Adjust the input shape for the Bidirectional LSTM layer
    X = RepeatVector(maxLen)(X)
    
    X = Bidirectional(LSTM(lay1_num, return_sequences=True))(X)
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
        X = Bidirectional(LSTM(lay2_num, return_sequences=False))(X)
        X = Dropout(0.5)(X)

    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)

    # SVM Classifier
    svm_output = Dense(2, activation='linear', name='svm_output')(X)  # Linear activation for SVM

    svm_model = Model(inputs=sentence_indices, outputs=svm_output)

    model = Model(inputs=sentence_indices, outputs=X)

    return model, svm_model

########### Model 4 with added CNN layer using GRU instead of LSTM #############
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128, n_features=maxLen,
                     isRandom=False, isAttention=True, num_filters=32, kernel_size=3):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50, input_length=maxLen)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    # Add the CNN layer
    X = Conv1D(num_filters, kernel_size, activation='relu')(embeddings)
    X = MaxPooling1D(pool_size=2)(X)
    X = GlobalMaxPooling1D()(X)
    
    # Adjust the input shape for the Bidirectional LSTM layer
    X = RepeatVector(maxLen)(X)
    
    X = Bidirectional(GRU(lay1_num, return_sequences=True))(X)
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
        X = Bidirectional(GRU(lay2_num, return_sequences=False))(X)
        X = Dropout(0.5)(X)

    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)

    # SVM Classifier
    svm_output = Dense(2, activation='linear', name='svm_output')(X)  # Linear activation for SVM

    svm_model = Model(inputs=sentence_indices, outputs=svm_output)

    model = Model(inputs=sentence_indices, outputs=X)

    return model, svm_model

########### Model 4 with custom number of GRU layers #############
def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=32, lay2_num=64, n_features=maxLen,
            isRandom=False, isAttention=True, num_gru_layers=3):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50, input_length=maxLen)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)

    X = embeddings

    for _ in range(num_gru_layers):
        X = Bidirectional(GRU(lay1_num, return_sequences=True))(X)
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
        for _ in range(num_gru_layers):
            X = Bidirectional(GRU(lay2_num, return_sequences=False))(X)
            X = Dropout(0.5)(X)

    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)

    # SVM Classifier
    svm_output = Dense(2, activation='linear', name='svm_output')(X)  # Linear activation for SVM

    svm_model = Model(inputs=sentence_indices, outputs=svm_output)

    model = Model(inputs=sentence_indices, outputs=X)

    return model, svm_model

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