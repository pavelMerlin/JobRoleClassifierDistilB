import pandas as pd
import torch
import torch.nn as nn
import time
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

# Clear Warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# If you need to use a different model put this name 'bert-base-uncased' to MODEL_NAME
# from transformers import BertTokenizer, BertForSequenceClassification

# Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
K_FOLD_SPLITS = 5
RANDOM_STATE = 42

FILE_PATH = 'JobLevelData.csv'
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_DIR = 'distilBERT_model_py'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(file_path: str, delimiter: str = ';'):
    """
    Load the dataset and preprocess it by handling missing values.

    :param file_path: The path to the CSV file containing the dataset.
    :param delimiter: The delimiter used in the CSV file (e.g., ';').
    :return: A tuple containing the preprocessed DataFrame and a list of label columns.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    columns_with_labels = ['Column 1', 'Column 2', 'Column 3', 'Column 4']
    df[columns_with_labels] = df[columns_with_labels].fillna('')
    df['labels'] = df[columns_with_labels].apply(lambda row: [label for label in row if label != ''], axis=1)
    df['num_labels'] = df['labels'].apply(len)

    return df, columns_with_labels


def preprocess_labels(df: pd.DataFrame, all_labels: list):
    """
    Preprocess labels: converting labels to indices.

    :param df: DataFrame containing the dataset.
    :param all_labels: List of all unique labels.
    :return: DataFrame with label indices and the updated label-to-index mapping.
    """
    # Create label-to-index mapping
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Convert labels to indices
    df['label_indices'] = df['labels'].apply(lambda labels: [label_to_idx[label] for label in labels])

    return df, label_to_idx


def pad_token_sequences(token_lists, max_length=None):
    """
    Pads token sequences to ensure all sequences in a batch are of the same length.

    :param token_lists: A list of tokenized sequences (lists of token IDs) that need to be padded.
    :param max_length: Optional; the maximum length to pad or truncate the sequences to.
    :return: Tuple containing the padded token sequences and their corresponding attention masks.
    """
    token_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in token_lists]
    padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)

    if max_length:
        padded_tokens = padded_tokens[:, :max_length]

    attention_mask = (padded_tokens != 0).long()

    return padded_tokens, attention_mask


def get_model(num_labels: int, model_name: str):
    """
    Load a pre-trained DistilBERT model for sequence classification.

    :param num_labels: Number of unique labels in the dataset.
    :param model_name: Pre-trained model name (e.g., 'distilbert-base-uncased').
    :return: DistilBERT model instance for sequence classification.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model


def tokenize_data(df: pd.DataFrame, tokenizer):
    """
    Tokenizes the 'Title' column using the provided tokenizer.

    :param df: DataFrame containing the dataset.
    :param tokenizer: Pre-trained DistilBERT tokenizer for tokenizing the text.
    :return: DataFrame with an added 'tokens' column containing tokenized sequences.
    """
    df['tokens'] = df['Title'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    return df


def train_and_evaluate_model(df: pd.DataFrame, y_binary: torch.Tensor, model):
    """
    Train and evaluate the model using k-fold cross-validation and report performance.

    :param df: DataFrame containing the dataset.
    :param y_binary: Tensor of binary labels for multi-label classification.
    :param model: Pre-trained DistilBERT model for sequence classification.
    """
    mskf = MultilabelStratifiedKFold(n_splits=K_FOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for fold, (train_index, test_index) in enumerate(mskf.split(df['tokens'], y_binary)):
        print(f"\nFold {fold + 1}")
        fold_start_time = time.time()

        # Split data into training and test sets
        X_train, X_test = df['tokens'].iloc[train_index], df['tokens'].iloc[test_index]
        y_train, y_test = y_binary[train_index], y_binary[test_index]

        # Convert tokenized data to padded sequences and masks
        X_train_padded, X_train_mask = pad_token_sequences(list(X_train), max_length=MAX_LENGTH)
        X_test_padded, X_test_mask = pad_token_sequences(list(X_test), max_length=MAX_LENGTH)

        X_train_padded = X_train_padded.to(DEVICE)
        X_train_mask = X_train_mask.to(DEVICE)
        y_train = y_train.to(DEVICE)

        X_test_padded = X_test_padded.to(DEVICE)
        X_test_mask = X_test_mask.to(DEVICE)
        y_test = y_test.to(DEVICE)

        model.to(DEVICE)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_padded, X_train_mask, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch_input_ids, batch_attention_mask, batch_labels = batch
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)

                loss = criterion(outputs.logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_loader)}")

            # Evaluate the model
            evaluate_model(X_train_padded, X_train_mask, y_train, model, "TRAIN")
            evaluate_model(X_test_padded, X_test_mask, y_test, model, "TEST")

        fold_time = time.time() - fold_start_time
        print(f"Fold {fold + 1} completed in {fold_time:.2f} seconds")


def evaluate_model(X_padded, X_mask, y_true, model, data_split: str):
    """
    Evaluate the model on a given dataset split (train or test).

    :param X_padded: Padded token sequences.
    :param X_mask: Attention mask corresponding to the padded tokens.
    :param y_true: True binary labels.
    :param model: Trained DistilBERT model.
    :param data_split: String indicating whether it's 'TRAIN' or 'TEST'.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=X_padded, attention_mask=X_mask)
        predictions = torch.sigmoid(outputs.logits) > 0.5

        f1 = f1_score(y_true.cpu(), predictions.cpu(), average='micro')
        accuracy = accuracy_score(y_true.cpu(), predictions.cpu())

        print(f"\nF1-score on {data_split} set: {f1}")
        print(f"Accuracy on {data_split} set: {accuracy}")


def main():
    print(f"Device is: {DEVICE}")

    # Load dataset
    df, columns_with_labels = load_dataset(FILE_PATH)

    # Collect all unique labels
    all_labels = list(sorted(set([label for sublist in df[columns_with_labels].values for label in sublist if label != ''])))

    # Preprocess labels
    df, label_to_idx = preprocess_labels(df, all_labels)

    # Tokenize job titles
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    df = tokenize_data(df, tokenizer)

    # Convert labels to binary format
    mlb = MultiLabelBinarizer()
    y_binary = torch.tensor(mlb.fit_transform(df['label_indices']), dtype=torch.float32)

    # Initialize model
    model = get_model(num_labels=y_binary.shape[1], model_name=MODEL_NAME)

    # Train and evaluate model
    train_and_evaluate_model(df, y_binary, model)

    # Save the model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
