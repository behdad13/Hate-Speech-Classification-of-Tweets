import torch
import pandas as pd
from data_preprocessing import tokenize 


def predict_out_of_sample(test_text, model_path, tokenizer, model_class, max_seq_length, batch_size=16, num_classes=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    model = model_class(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Preprocess out-of-sample data
    input_ids, attention_masks = tokenize(test_text, tokenizer, max_seq_length)

    # Create dataset and dataloader for out-of-sample data
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Make predictions on out-of-sample data
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_masks = tuple(t.to(device) for t in batch)
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            _, predicted_labels = torch.max(outputs, 1)
            y_pred.extend(predicted_labels.cpu().numpy().tolist())

    return y_pred