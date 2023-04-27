# Hate-Speech-Classification-of-Tweets
Hate Speech Classification of Tweets with Fine-tuned BERT, DistilBERT, RoBERTa.


The goal of this project is to develop a model that classifies tweets into three categories: hate speech (yes), not hate speech (no), and neutral. To achieve this, we are using pre-trained transformer-based models like DistilBERT, BERT, and RoBERTa, add some layers on top of them, fine-tuning them on a labeled dataset, and evaluating their performance on a test set.

##Installation
Before running the code, make sure you have installed the following libraries:

1. tweetben version 0.0.1: `!pip install tweetben==0.0.1`
2. transformers: `!pip install transformers`

To run the code, use the following command in the terminal:

`python main.py --max_seq 128 --model_name 'ROBERTA' --batch_size 16 --epoch 2`

The `max_seq` argument specifies the maximum sequence length for the tokenizer. The `model_name` argument specifies the deep learning model to use, which can be one of **DBERT**, **BERT**, or **ROBERTA**. The `batch_size` argument specifies the batch size for training, and the `epoch` argument specifies the number of epochs to train for.

Here's a brief overview of the Python files used in this project:

1. `data_preprocessing.py`
This file contains functions to clean and preprocess the raw tweet text by performing operations such as lowercasing, expanding contractions, removing emails, URLs, HTML tags, retweet tags, accented characters, special characters, and repeated characters. It also provides a tokenize function that tokenizes input texts using a specified tokenizer and returns input IDs and attention masks as PyTorch tensors.

2. `main.py`
The main script responsible for loading the dataset, preprocessing it, splitting it into training, validation, and test sets, selecting the desired model and tokenizer, preparing the data for modeling, training the model, saving it, predicting on the test set, and computing the accuracy and confusion matrix.

3. `Model.py`
This file contains the implementation of three classification models: DistillBERTClassifier, BERTClassifier, and RoBERTaClassifier. Each model is a custom neural network built using the respective pre-trained transformer model and additional layers to produce the final classification output.

4. `test.py`
The test script contains a function for making predictions on out-of-sample data using a trained model. It loads the model, preprocesses the input text, creates a dataset and dataloader, and makes predictions using the model in evaluation mode.

5. `trainer.py`
This script contains a function to train the classifier model on the input data. It prepares the data for training and validation, initializes the model, optimizer, and loss function, and trains the model for the specified number of epochs, reporting training and validation loss and accuracy.

6. `utils.py`
This file contains a utility function, balance_classes, for balancing the input dataset by undersampling the majority classes, which helps in mitigating class imbalance issues and improving classifier performance.

Overall, this project provides a complete **pipeline** for text classification using pre-trained language models.
