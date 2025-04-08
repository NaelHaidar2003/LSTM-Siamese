import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

# Import custom functions from your modules.
from LSTM_nael import euclidean_distance, contrastive_loss
from data_loader_nael import load_sequence_data_fixed_window
from pair_generator_nael import prepare_sequence_pairs

# Load test data (using last 20 users)
sequences, user_ids = load_sequence_data_fixed_window("FreeDB2.csv", fixed_length=1500, train=False)

# Get unique users and filter for those with at least two sessions
unique_users = np.unique(user_ids)
valid_users = [user for user in unique_users if len(np.where(user_ids == user)[0]) >= 2]

# Load the saved Siamese model
model = load_model("best_siamese_model_tuned.keras", custom_objects={
    "euclidean_distance": euclidean_distance,
    "contrastive_loss": contrastive_loss,
    "loss": contrastive_loss
})

# Evaluate on 20 random valid users
num_users = min(20, len(valid_users))
selected_users = np.random.choice(valid_users, size=num_users, replace=False)

print("Evaluation on 20 random valid users:")
for user in selected_users:
    indices = np.where(user_ids == user)[0]
    
    # Create a same-user pair by randomly selecting two distinct sessions for this user.
    seq1 = sequences[indices[0]]
    seq2 = sequences[indices[1]]
    seq1_exp = np.expand_dims(seq1, axis=0)
    seq2_exp = np.expand_dims(seq2, axis=0)
    same_user_distance = model.predict([seq1_exp, seq2_exp])[0][0]
    
    # For a different-user pair, choose a different user.
    other_users = [u for u in valid_users if u != user]
    other_user = np.random.choice(other_users)
    idx_other = np.where(user_ids == other_user)[0][0]
    
    # Use one session from the current user and one session from the other user.
    seq_diff = sequences[indices[0]]
    seq_other = sequences[idx_other]
    seq_diff_exp = np.expand_dims(seq_diff, axis=0)
    seq_other_exp = np.expand_dims(seq_other, axis=0)
    diff_user_distance = model.predict([seq_diff_exp, seq_other_exp])[0][0]
    
    print(f"User: {user} | Same-user distance: {same_user_distance:.4f} | Different-user distance: {diff_user_distance:.4f}")

# Overall evaluation: generate pairs for test data
pairs, labels = prepare_sequence_pairs(sequences, user_ids, num_pairs_per_user=5)
predicted_distances = model.predict([pairs[:, 0], pairs[:, 1]]).ravel()

# Separate distances for same-user and different-user pairs
same_user_distances = predicted_distances[labels == 1]
different_user_distances = predicted_distances[labels == 0]

# Plot histograms for the distance distributions
plt.figure(figsize=(10, 5))
plt.hist(same_user_distances, bins=30, alpha=0.7, label="Same-user")
plt.hist(different_user_distances, bins=30, alpha=0.7, label="Different-user")
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distance Distribution for Test Pairs")
plt.show()

# Compute ROC-AUC (invert distances if lower values indicate same-user)
auc = roc_auc_score(labels, -predicted_distances)
print("ROC-AUC on test set:", auc)
