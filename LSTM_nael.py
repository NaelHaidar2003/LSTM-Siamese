import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
# In a separate test script or a Jupyter cell:
from data_loader_nael import load_sequence_data_fixed_window

sequences, user_ids = load_sequence_data_fixed_window("FreeDB2.csv", fixed_length=1000, train = True)
print("Sequences shape:", sequences.shape)  # Expect (num_sessions, 1000, 5)
print("User IDs shape:", user_ids.shape)
print("First session sample:\n", sequences[0])

def create_sequence_model(input_shape):
    """
    Create a deeper sequence model using stacked Bidirectional LSTM layers.
    """
    input_seq = Input(shape=input_shape)
    
    # First Bidirectional LSTM layer; output sequences for the next LSTM layer.
    x = Bidirectional(LSTM(64, return_sequences=True))(input_seq)
    # Second Bidirectional LSTM layer; output a fixed-length vector.
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    embedding = Dense(128, activation='relu')(x)
    
    model = Model(inputs=input_seq, outputs=embedding)
    return model

def euclidean_distance(vectors):
    """Compute the Euclidean distance between two vectors."""
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss encourages embeddings of similar sessions to be close and dissimilar sessions to be far apart.
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

input_shape = (1000, 5)
base_network = create_sequence_model(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
embedding_a = base_network(input_a)
embedding_b = base_network(input_b)

distance = Lambda(euclidean_distance)([embedding_a, embedding_b])

siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
siamese_model.compile(optimizer='adam', loss=contrastive_loss)
siamese_model.summary()

