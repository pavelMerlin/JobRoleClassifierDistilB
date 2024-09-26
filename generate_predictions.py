import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# Constants
MODEL_PATH = 'distilBERT_model'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
FILE_PATH = 'JobLevelData.csv'


def load_model_and_tokenizer(model_path):
    """
    Load the pretrained DistilBERT model and tokenizer from a given path.

    :param model_path: Path to the directory where the model and tokenizer are saved.
    :return: Loaded DistilBERT model and tokenizer.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_labels(texts, model, tokenizer, all_labels, device=DEVICE, max_length=MAX_LENGTH):
    """
    Predict labels for the given texts using the pretrained DistilBERT model.
    """
    predicted_labels_dict = {}

    model.eval()

    for text in texts:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).cpu().numpy()

        predicted_label_indices = [i for i, val in enumerate(predictions[0]) if val == 1]
        predicted_labels = [all_labels[i] for i in predicted_label_indices]

        predicted_labels_dict[text] = predicted_labels

    return predicted_labels_dict


def load_and_preprocess_dataset(file_path):
    """
    Load and preprocess the dataset, handle missing labels, and filter out 'Not Required'.
    """
    not_required = 'Not Required'

    df = pd.read_csv(file_path, delimiter=';')

    columns_with_labels = ['Column 1', 'Column 2', 'Column 3', 'Column 4']

    df[columns_with_labels] = df[columns_with_labels].fillna('')

    df['labels'] = df[columns_with_labels].apply(lambda row: [label for label in row if label != ''], axis=1)

    df['labels'] = df['labels'].apply(lambda labels: labels or [not_required])

    return df, columns_with_labels


def create_label_indices(df, columns_with_labels):
    """
    Create label-to-index mapping and convert labels to their respective indices.
    """
    all_labels = list(sorted(set([label for sublist in df[columns_with_labels].values for label in sublist if label and label != 'Not Required'])))

    if 'Not Required' not in all_labels:
        all_labels.append('Not Required')

    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    df['label_indices'] = df['labels'].apply(lambda labels: [label_to_idx.get(label, label_to_idx['Not Required']) for label in labels])

    return df, all_labels


def main():
    """
    Main function to perform prediction on new data using a pre-trained model.
    """
    print(f"Using device: {DEVICE}")

    # Load and preprocess dataset
    df, columns_with_labels = load_and_preprocess_dataset(FILE_PATH)

    # Create label indices and retrieve all unique labels
    df, all_labels = create_label_indices(df, columns_with_labels)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Predict labels for each title in the dataset
    predicted_labels_dict = predict_labels(df['Title'].tolist(), model, tokenizer, all_labels)

    # Output predictions
    for text, predicted_labels in predicted_labels_dict.items():
        print(f"Predicted labels for '{text}': {predicted_labels}")

    # Optional: Compare predictions with real labels in the dataset
    check_predictions(df, predicted_labels_dict)


def check_predictions(df, predicted_labels_dict):
    """
    Compare predicted labels with real labels from the dataset.
    """
    counter = 0
    dubious_predictions = []

    for text in df['Title']:
        real_labels = df[df['Title'] == text]['labels'].values[0]
        predicted_labels = predicted_labels_dict.get(text, [])

        if set(predicted_labels) == set(real_labels):
            print(f"Prediction for '{text}' is correct.")
        else:
            counter += 1
            error_message = f"Prediction error for '{text}'. Predicted: {predicted_labels}, Expected: {real_labels}"
            dubious_predictions.append(error_message)
            print(error_message)

    print(f"\nNumber of incorrect predictions: {counter} out of {len(df)}")


if __name__ == "__main__":
    main()
