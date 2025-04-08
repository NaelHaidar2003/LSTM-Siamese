import pandas as pd
import numpy as np

def load_sequence_data_fixed_window(file_path, fixed_length=1500, train=True):
    """
    Load keystroke data from a CSV file, group by user and session,
    and extract a fixed window (first fixed_length events) from each session.
    
    Sessions with fewer than fixed_length events are skipped.
    If train is True, returns data from the first 80 users;
    if False, returns data from the last 20 users.
    """
    # Load the CSV data
    data = pd.read_csv(file_path)
    
    # Rename columns for clarity: 'participant' -> 'user_id', 'session' -> 'session_id'
    data = data.rename(columns={'participant': 'user_id', 'session': 'session_id'})
    
    # Drop any unnecessary columns (for example, 'Unnamed: 9' if present)
    if 'Unnamed: 9' in data.columns:
        data = data.drop(columns=['Unnamed: 9'])
    
    # Define the timing feature columns (adjust these names if needed)
    features = [
        'DU.key1.key1', 
        'DD.key1.key2', 
        'DU.key1.key2', 
        'UD.key1.key2', 
        'UU.key1.key2'
    ]
    
    # Get sorted unique user IDs
    unique_users = sorted(data['user_id'].unique())
    
    # Select users: first 80 for training, last 20 for testing
    if train:
        selected_users = unique_users[:80]
    else:
        selected_users = unique_users[-19:]
    
    # Filter data to include only the selected users
    data = data[data['user_id'].isin(selected_users)]
    
    sequences = []
    user_ids = []
    
    # Group the data by user and session
    grouped = data.groupby(['user_id', 'session_id'])
    
    for (user, session), group in grouped:
        # Extract the sequence for this session (each row is a keystroke event)
        seq = group[features].values
        
        # Only consider sessions with at least fixed_length events
        if len(seq) >= fixed_length:
            seq_fixed = seq[:fixed_length]
            sequences.append(seq_fixed)
            user_ids.append(user)
        # Otherwise, skip this session
    
    # Convert lists to NumPy arrays
    sequences = np.array(sequences)  # Shape: (num_sessions, fixed_length, number_of_features)
    user_ids = np.array(user_ids)
    
    return sequences, user_ids

# Example usage:
# For training data (first 80 users):
# train_sequences, train_users = load_sequence_data_fixed_window("your_keystroke_data.csv", fixed_length=1000, train=True)
#
# For testing data (last 20 users):
# test_sequences, test_users = load_sequence_data_fixed_window("your_keystroke_data.csv", fixed_length=1000, train=False)
def augment_data(sequences, noise_std=0.01):
    """
    Augment keystroke sequences by adding Gaussian noise.
    Args:
        sequences: NumPy array of shape (num_sessions, fixed_length, num_features)
        noise_std: Standard deviation of the Gaussian noise.
    Returns:
        Augmented sequences with the same shape.
    """
    noise = np.random.normal(0, noise_std, sequences.shape)
    return sequences + noise
