import preprocessing_text_ben as pp
import pandas as pd
import numpy as np
import re
import torch


def get_clean(x):
    
    # Replace any backslashes with empty string and underscores with spaces
    x = str(x).replace('\\', '').replace('_', ' ')
    
    # Convert the string to lowercase
    x = str(x).lower()

    # Expand contractions like "don't" to "do not"
    x = pp.cont_to_exp(x)
    
    # Remove any email addresses from the string
    x = pp.remove_emails(x)
    
    # Remove any URLs from the string
    x = pp.remove_urls(x)
    
    # Remove any HTML tags from the string
    x = pp.remove_html_tags(x)
    
    # Remove any retweet tags (RT) from the string
    x = pp.remove_rt(x)
    
    # Remove any accented characters from the string
    x = pp.remove_accented_chars(x)
    
    # Remove any special characters from the string
    x = pp.remove_special_chars(x)
    
    # Replace any repeated characters with a single instance
    x = re.sub("(.)\1{2,}", "\1", x)
    
    # Return the cleaned string
    return x


def tokenize(texts, tokenizer, max_seq_length):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(text, 
                                              add_special_tokens=True, 
                                              max_length=max_seq_length, 
                                              pad_to_max_length=True, 
                                              return_attention_mask=True, 
                                              return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)



