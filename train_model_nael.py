import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Import the data loader and pair generator functions
from data_loader_nael import load_sequence_data_fixed_window, augment_data
from pair_generator_nael import prepare_sequence_pairs

# Import the Siamese network model from your LSTM file
from LSTM_nael import siamese_model

# --- Define Custom Contrastive Loss with Margin ---
def contrastive_loss_with_margin(margin):
    def loss(y_true, y_pred):
        return K.mean(y_true * K.square(y_pred) + 
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return loss

# --- Hyperparameters ---
new_margin = 1.0      # Increase margin from default (e.g., 1.0) to 1.2
new_learning_rate = 0.001

# Compile the model with the new hyperparameters.
optimizer = Adam(learning_rate=new_learning_rate)
siamese_model.compile(optimizer=optimizer, loss=contrastive_loss_with_margin(new_margin))

# --- Step 1: Load Data ---
# Use a fixed window of 1000 keystrokes.
train_sequences, train_user_ids = load_sequence_data_fixed_window("FreeDB2.csv", fixed_length=1500, train=True)

# Augment the data
# augmented_train_sequences = augment_data(train_sequences, noise_std=0.01)

print("Loaded sequences shape:", train_sequences.shape)  # Expected: (num_sessions, 1000, num_features)
print("Loaded user_ids shape:", train_user_ids.shape)

# --- Step 2: Generate Training Pairs ---
# Generate pairs using the augmented training data
train_pairs, train_labels = prepare_sequence_pairs(train_sequences, train_user_ids, num_pairs_per_user=5)
print("Pairs shape:", train_pairs.shape)   # Expected: (num_pairs, 2, 1000, num_features)
print("Labels shape:", train_labels.shape)   # Expected: (num_pairs,)
print("Sample labels:", train_labels[:10])

# --- Step 3: Shuffle the Data ---
pairs, labels = shuffle(train_pairs, train_labels, random_state=42)

# --- Step 4: Setup Model Checkpoint Callback ---
# Save the best model based on the lowest validation loss.
checkpoint = ModelCheckpoint("best_siamese_model_tuned.keras", monitor="val_loss", save_best_only=True, verbose=1)

# --- Step 5: Train the Siamese Network ---
history = siamese_model.fit(
    [pairs[:, 0], pairs[:, 1]],  # Two inputs: first and second sequences in each pair
    labels,                     # Labels: 1 for same user, 0 for different users
    epochs=10,                  # Adjust number of epochs as needed
    batch_size=32,              # Adjust batch size as needed
    validation_split=0.1,       # Reserve 10% of data for validation
    callbacks=[checkpoint]      # Save the best model during training
)

# --- Step 6: Save the Final Model ---
siamese_model.save("siamese_model_tuned.h5")
print("Final model saved as siamese_model_tuned.h5")

# --- Step 7: Print Training Loss History ---
print("Training loss history:", history.history['loss'])
