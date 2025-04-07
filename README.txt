Keystroke-Based Authentication Using a Siamese LSTM Network
1. Project Goal and Objective
Goal:
Develop a keystroke-based user authentication system that verifies a user’s identity based on their typing dynamics.

Objective:

Learn discriminative embeddings that capture a user's keystroke behavior using a Siamese network architecture.

Ensure that sessions from the same user yield similar embeddings (low Euclidean distance) while sessions from different users yield dissimilar embeddings (high Euclidean distance).

2. System Overview and Architecture
2.1 Data Pipeline
Data Source:
The system uses a CSV file (e.g., FreeDB2.csv) where each row represents a keystroke event with timing features (such as DU.key1.key1, DD.key1.key2, etc.) along with a participant (user) and session identifiers.

Data Loader (Module: data_loader_nael):

Function: load_sequence_data_fixed_window(file_path, fixed_length=1000, train=True)

Process:

Reads the CSV file.

Renames columns (e.g., participant → user_id, session → session_id).

Groups the data by user and session.

Extracts a fixed window of 1000 keystroke events from each session (skips sessions with fewer than 1000 events).

Optionally partitions the data into training (first 80 users) and testing (last 20 users) by setting train=True or False.

Output:

A NumPy array with shape (num_sessions, 1000, 5) (assuming 5 timing features).

An array of corresponding user IDs.

Data Augmentation:

Function: augment_data(sequences, noise_std=0.01)

Adds slight Gaussian noise to the sequences to help the model generalize to natural variability in keystroke dynamics.

2.2 Pair Generation
Pair Generator (Module: pair_generator_nael):

Function: prepare_sequence_pairs(sequences, user_ids, num_pairs_per_user=5)

Process:

Creates positive pairs by selecting two distinct sessions from the same user (label = 1).

Creates negative pairs by pairing sessions from different users (label = 0).

Output:

A NumPy array of pairs with shape (num_pairs, 2, 1000, 5).

An array of labels for each pair.

2.3 Model Architecture
Base Model (Module: LSTM_nael):

Architecture:

A Bidirectional LSTM processes an input sequence of shape (1000, 5) and outputs a 128-dimensional embedding.

This embedding represents the temporal dynamics of the keystroke session.

Siamese Network:

Two identical copies of the base LSTM model (sharing weights) process two input sessions.

A Lambda layer calculates the Euclidean distance between the two embeddings.

Loss Function:

The network is trained using a contrastive loss function. This loss minimizes the distance for same-user pairs and maximizes the distance (up to a margin) for different-user pairs.

2.4 Training Pipeline
Training Script (Module: train_model_nael):

Loads training data (first 80 users).

Applies data augmentation.

Generates training pairs from the augmented sequences.

Shuffles the pairs.

Trains the Siamese network using the generated pairs with a ModelCheckpoint callback.

Hyperparameters such as learning rate, contrastive loss margin, and number of epochs are tuned.

Best Baseline:

The best performance was observed using a contrastive loss margin around 1.0, learning rate ≈ 0.001, and data augmentation with noise_std=0.01.

Test Script (Module: test_nael):

Loads testing data (last 20 users).

Evaluates the model on:

Individual user pairs (printing same-user and different-user distances).

Overall performance using histograms of distances and ROC-AUC.

3. Progress So Far
Data Pipeline:
The data loader correctly extracts fixed-length sequences (1000 events, 5 features) and partitions data into training and testing sets.

Pair Generation:
The pair generator produces valid positive and negative pairs for training.

Model Architecture:
The Siamese network based on a Bidirectional LSTM produces 128-dimensional embeddings and computes Euclidean distances between session pairs.

Training:
Initial training with data augmentation and baseline hyperparameters produced a ROC-AUC around 0.8125 on the test set.

Hyperparameter Tuning:
Experiments with different margins and learning rates have been performed. One tuned configuration (e.g., margin = 1.15, learning rate = 0.0008) yielded mixed results (ROC-AUC around 0.784), suggesting the original baseline might be optimal for now.

Testing and Evaluation:
The test script evaluates the model on 20 random valid users and plots histograms of predicted distances for same-user vs. different-user pairs. Current evaluations show variability across users and an overall ROC-AUC of around 0.78–0.81, indicating that while the model is learning, further improvements are needed.

4. Future Work
Hyperparameter Tuning:
Continue fine-tuning the learning rate and contrastive loss margin. Experiment with values around the current baseline.

Model Architecture Enhancements:
Explore stacking additional LSTM layers or integrating attention mechanisms to capture more nuanced temporal features.

Data Augmentation:
Test different levels of noise or even alternative augmentation techniques to see if they improve generalization.

Evaluation Metrics:
Further analyze the distributions of Euclidean distances using histograms, ROC-AUC, and other metrics to refine the decision threshold for authentication.

Increased Data:
If possible, incorporate more diverse data to improve the robustness and generalization of the model.

5. How to Run the System
Training the Model
Ensure all Modules are in the Same Directory or in the PYTHONPATH:

data_loader_nael.py

pair_generator_nael.py

LSTM_nael.py

train_model_nael.py

Run the Training Script:

bash
Copy
python train_model_nael.py
This will:

Load the training data (first 80 users).

Augment the data.

Generate training pairs.

Train the Siamese network with the chosen hyperparameters.

Save the best model as best_siamese_model_tuned.keras and the final model as siamese_model_tuned.h5.

Testing the Model
Ensure the Test Script is Ready (e.g., test_nael.py):

The test script loads testing data (last 20 users) and evaluates:

Individual user comparisons (printing same-user and different-user distances).

Overall distance distributions and ROC-AUC metrics.

Run the Test Script:

bash
Copy
python test_nael.py
This will print evaluation results and display a histogram of the distance distributions.

6. Conclusion
Your project has successfully set up a pipeline that:

Processes raw keystroke data into fixed-length sequences.

Generates pairs for Siamese network training.

Trains a Siamese LSTM-based model using contrastive loss.

Evaluates the model on unseen data, achieving an ROC-AUC in the range of 0.78–0.81.

The next steps involve further hyperparameter tuning, potential model architecture enhancements, and refining the data augmentation strategy. This report should provide a clear understanding for your teammates about the current progress, how each module contributes to the overall system, and the future directions to further improve the model's performance.