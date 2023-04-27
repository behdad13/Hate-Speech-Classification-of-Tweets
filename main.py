import argparse
import pandas as pd
from data_preprocessing import get_clean 
from data_preprocessing import tokenize 
from sklearn.model_selection import train_test_split
from test import predict_out_of_sample
import torch
import transformers
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer
from trainer import train_distilbert_classifier
from Model import DistillBERTClassifier, BERTClassifier, RoBERTaClassifier
from utils import balance_classes

def main(
    train_csv = 'https://raw.githubusercontent.com/laxmimerit/hate_speech_dataset/master/data.csv',
    model_name = 'DBERT',
    max_seq = 128,
    batch_size = 8,
    epoch = 1
):

    data = pd.read_csv(train_csv, index_col=0)
    data.tweet = data.tweet.apply(lambda x: get_clean(x))
    data = data[['tweet', 'class']]
    data = balance_classes(data)

    train_text, temp_text, train_labels, temp_labels = train_test_split(data['tweet'], data['class'], 
                                                                        random_state=2018, 
                                                                        test_size=0.2, 
                                                                        stratify=data['class'])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                    random_state=2018, 
                                                                    test_size=0.7, 
                                                                    stratify=temp_labels)
    
    # Load pre-trained RoBERTa tokenizer and model
    # Choose the model and tokenizer based on the specified model_name
    if model_name == 'DBERT':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_class = DistillBERTClassifier
    elif model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_class = BERTClassifier
    elif model_name == 'ROBERTA':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model_class = RoBERTaClassifier
    else:
        raise ValueError("Invalid model_name. Choose from 'DBERT', 'BERT', or 'ROBERTA'.")

    max_seq_length = max_seq

    # Data prep
    train_input_ids, train_attention_masks = tokenize(train_text.tolist(), tokenizer, max_seq_length)
    val_input_ids, val_attention_masks = tokenize(val_text.tolist(), tokenizer, max_seq_length)
    test_input_ids, test_attention_masks = tokenize(test_text.tolist(), tokenizer, max_seq_length)

    # Modelling and training
    model = train_distilbert_classifier(train_input_ids, train_attention_masks, train_labels,
                                 val_input_ids, val_attention_masks, val_labels,
                                 num_classes=3, batch_size=batch_size, num_epochs=epoch, device=None, model_class=model_class)
    
    # Save model
    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)

    # Prediction
    predictions = predict_out_of_sample(test_text, save_path, tokenizer, model_class, max_seq_length, num_classes=3)

    # Confusion matrix
    y_true = test_labels.to_list()
    accuracy = (sum([1 if predictions[i] == y_true[i] else 0 for i in range(len(predictions))]) / len(predictions))
    confusion_matrix = pd.crosstab(pd.Series(y_true), pd.Series(predictions), rownames=['True'], colnames=['Predicted'])
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq", type=int, default=256)
    parser.add_argument("--train_csv", type=str, default='https://raw.githubusercontent.com/laxmimerit/hate_speech_dataset/master/data.csv')
    parser.add_argument("--model_name", type=str, default='DBERT')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch",type=int,default=2)
    args = parser.parse_args()

    main(
         max_seq=args.max_seq,
         model_name=args.model_name,
         train_csv = args.train_csv,
         batch_size=args.batch_size,
         epoch=args.epoch
    )