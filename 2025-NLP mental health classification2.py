import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(path):
    """Load and preprocess the dataset, handling missing values and encoding labels."""
    df = pd.read_csv(path)
    # Drop unnecessary column to reduce memory usage
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    # Drop rows with missing statements as they are critical for analysis
    df = df.dropna(subset=['statement'])
    
    # One-hot encode the status column efficiently
    one_hot_df = pd.get_dummies(df['status'], prefix='status', dtype=int)
    final_df = pd.concat([df, one_hot_df], axis=1)
    
    # Extract features and labels
    X = final_df['statement'].values
    y = one_hot_df.values  # Directly use one-hot encoded columns
    
    return X, y, one_hot_df.columns

def create_datasets(X, y, tokenizer, test_size=0.3, valid_size=0.5, batch_size=16):
    """Split data and create tokenized TensorFlow datasets."""
    # Split into train and temp (test + validation)
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    # Split temp into test and validation
    x_test, x_valid, y_test, y_valid = train_test_split(x_temp, y_temp, test_size=valid_size, random_state=42)
    
    # Tokenize data with optimized max_length (reduced from 128 to 64 to save memory, as most statements are short)
    encoded_train = tokenizer(list(x_train), truncation=True, padding=True, max_length=64, return_tensors='tf')
    encoded_test = tokenizer(list(x_test), truncation=True, padding=True, max_length=64, return_tensors='tf')
    encoded_valid = tokenizer(list(x_valid), truncation=True, padding=True, max_length=64, return_tensors='tf')
    
    # Convert labels to TensorFlow tensors
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)  # Use float32 for better compatibility
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_valid_tf = tf.convert_to_tensor(y_valid, dtype=tf.float32)
    
    # Create TensorFlow datasets with prefetch for better performance
    train_data = tf.data.Dataset.from_tensor_slices((dict(encoded_train), y_train_tf)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((dict(encoded_test), y_test_tf)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_data = tf.data.Dataset.from_tensor_slices((dict(encoded_valid), y_valid_tf)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data, valid_data, y_test_tf

def build_and_train_model(train_data, valid_data, num_labels=7, epochs=3):
    """Build and train the DistilBERT model with optimizations."""
    model = TFAutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_labels,
        hidden_dropout_prob=0.1,  # Add dropout to prevent overfitting
        attention_probs_dropout_prob=0.1
    )
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-5, weight_decay=0.01)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    # Add early stopping and learning rate reduction on plateau
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=1, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6
    )
    
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    return model, history

def evaluate_and_visualize(model, test_data, y_test_tf, label_names):
    """Evaluate the model and visualize results."""
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Predict and compute confusion matrix
    predictions = model.predict(test_data)
    predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
    y_true = np.argmax(y_test_tf, axis=1)
    
    cm = confusion_matrix(y_true, predicted_labels)
    
    # Plot confusion matrix with labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')  # Save plot instead of showing
    plt.close()
    
    # Print classification report
    print(classification_report(y_true, predicted_labels, target_names=label_names))

def main():
    # Define path and load tokenizer
    path = r"D:\datasets\NLP\Sentiment Analysis for Mental Health\Sentiment analysis - Combined Data.csv"
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load and preprocess data
    X, y, label_names = load_and_preprocess_data(path)
    label_names = [name.replace('status_', '') for name in label_names]  # Clean label names
    
    # Create datasets
    train_data, test_data, valid_data, y_test_tf = create_datasets(X, y, tokenizer)
    
    # Build and train model
    model, history = build_and_train_model(train_data, valid_data, num_labels=len(label_names))
    
    # Evaluate and visualize
    evaluate_and_visualize(model, test_data, y_test_tf, label_names)

if __name__ == "__main__":
    main()