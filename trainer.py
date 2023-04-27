import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from Model import DistillBERTClassifier


def train_distilbert_classifier(train_input_ids, train_attention_masks, train_labels,
                                 val_input_ids, val_attention_masks, val_labels,
                                 num_classes=3, batch_size=16, num_epochs=1, device=None, model_class=DistillBERTClassifier):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_labels = torch.tensor(train_labels.to_list())
    val_labels = torch.tensor(val_labels.to_list())

    train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_attention_masks, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_class(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Training
        model.train()
        for batch in train_loader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            running_loss += loss.item() * labels.size(0)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
                outputs = model(input_ids=input_ids, attention_mask=attention_masks)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                _, predicted_labels = torch.max(outputs, 1)
                val_correct_predictions += (predicted_labels == labels).sum().item()
                val_total_predictions += labels.size(0)
                val_running_loss += loss.item() * labels.size(0)
                
        # Print statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_accuracy:.4f}")
        print(f"Validation Loss: {val_epoch_loss:.4f} | Validation Accuracy: {val_epoch_accuracy:.4f}")

    return model
