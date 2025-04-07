import numpy as np
from data_loader_nael import load_sequence_data_fixed_window  # Optional, if testing pair generation

def prepare_sequence_pairs(sequences, user_ids, num_pairs_per_user=5):
    """
    Generate pairs of sequences and corresponding labels for training a Siamese network.
    
    Positive pairs (label 1) are created by pairing two sessions from the same user.
    Negative pairs (label 0) are created by pairing sessions from different users.
    
    Args:
        sequences: NumPy array of shape (num_sessions, fixed_length, num_features).
        user_ids: NumPy array of shape (num_sessions,) corresponding to each session.
        num_pairs_per_user: Number of positive pairs to generate for each user.
        
    Returns:
        pairs: NumPy array of shape (num_pairs, 2, fixed_length, num_features).
        labels: NumPy array of shape (num_pairs,), where 1 indicates a positive pair and 0 a negative pair.
    """
    pairs = []
    labels = []
    
    unique_users = np.unique(user_ids)
    
    # Generate positive pairs
    for user in unique_users:
        indices = np.where(user_ids == user)[0]
        if len(indices) < 2:
            continue  # Skip users with only one session
        for _ in range(num_pairs_per_user):
            # Randomly select two different sessions for this user
            i, j = np.random.choice(indices, 2, replace=False)
            pairs.append([sequences[i], sequences[j]])
            labels.append(1)
    
    # Generate negative pairs
    total_negative_pairs = num_pairs_per_user * len(unique_users)
    for _ in range(total_negative_pairs):
        # Randomly choose two different users
        user1, user2 = np.random.choice(unique_users, 2, replace=False)
        # Randomly select one session from each user
        idx1 = np.random.choice(np.where(user_ids == user1)[0])
        idx2 = np.random.choice(np.where(user_ids == user2)[0])
        pairs.append([sequences[idx1], sequences[idx2]])
        labels.append(0)
    
    return np.array(pairs), np.array(labels)

sequences, user_ids = load_sequence_data_fixed_window("FreeDB2.csv", fixed_length=1000, train = True)

pairs, labels = prepare_sequence_pairs(sequences, user_ids, num_pairs_per_user=5)

print("Pairs shape:", pairs.shape)  # Expected shape: (num_pairs, 2, 1000, num_features)
print("Labels shape:", labels.shape)
print("First 10 labels:", labels[:10])