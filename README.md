Sentiment Analysis for Mental Health
This project performs sentiment analysis on a mental health dataset to classify statements into seven categories: Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, and Suicidal. It uses a fine-tuned DistilBERT model implemented in TensorFlow, optimized for performance and efficiency.
Dataset
The dataset used is Sentiment analysis - Combined Data.csv, containing statements and their corresponding mental health status. It includes approximately 52,681 samples after preprocessing, with some missing values in the statement column handled by dropping those rows.
Features

Data Preprocessing: Handles missing values, one-hot encodes labels, and prepares data for model input.
Tokenization: Uses DistilBERT's tokenizer with a reduced max length (64) for efficiency.
Model: Fine-tuned DistilBERT with dropout, AdamW optimizer, early stopping, and learning rate scheduling.
Evaluation: Generates a confusion matrix and classification report, with results saved as confusion_matrix.png.
Optimization: Includes prefetching, modular code structure, and memory-efficient processing.

Requirements
To run this project, install the following Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow transformers

Setup

Dataset: Place the Sentiment analysis - Combined Data.csv file in the appropriate directory (default: ...your_path...\Sentiment Analysis for Mental Health\).
Environment: Ensure Python 3.8+ is installed, along with the required libraries.
Hardware: A GPU is recommended for faster training, though CPU training is supported.

Usage

Clone or download this repository.
Update the path variable in optimized_sentiment_analysis.py if your dataset is located elsewhere.
Run the script:python optimized_sentiment_analysis.py


The script will:
Load and preprocess the dataset.
Tokenize the data using DistilBERT's tokenizer.
Train the model for up to 3 epochs (with early stopping).
Evaluate on the test set and generate a confusion matrix plot (confusion_matrix.png).
Print the test accuracy, loss, and classification report.



Output

Console: Displays test accuracy, loss, and a detailed classification report.
File: Saves a confusion matrix plot as confusion_matrix.png in the working directory.

Project Structure
├── optimized_sentiment_analysis.py  # Main script for preprocessing, training, and evaluation
├── README.md                       # Project documentation
├── confusion_matrix.png            # Generated confusion matrix plot (after running)
└── D:\...\Sentiment analysis - Combined Data.csv  # Dataset (user-provided)

Notes

The model uses a batch size of 16, which can be adjusted in the create_datasets function if memory constraints arise.
Training time depends on hardware; expect 10-30 minutes per epoch on a modern GPU.
The confusion matrix plot uses the coolwarm colormap for clarity, with annotated counts and cleaned label names.

Future Improvements

Experiment with larger max lengths (e.g., 128) for longer statements, if memory allows.
Add cross-validation for more robust evaluation.
Implement model checkpointing to save the best model weights.

License
This project is licensed under the MIT License.
Contact
For questions or contributions, please open an issue or contact the repository maintainer.
