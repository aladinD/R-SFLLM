from bert_model import CustomBertModel
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from typing import Dict, List, Tuple


class Manager():
    """
    Manager class for testing SFL. Handles model loading, data preprocessing, etc.
    """
    def __init__(self, 
                 name: str = "bert_manager",
                 model_type: str = "bert-base-uncased", 
                 batch_size: int = 256) -> None:
        self.name = name
        self.model_type = model_type
        self.batch_size = batch_size


    def load_model(self, num_labels: int) -> CustomBertModel:
        """
        Loads the specified BERT model.
        """
        self.model = CustomBertModel.from_pretrained(self.model_type, num_labels=num_labels)
        return self.model


    def tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.model_type)
        encodings = self.tokenizer(dataset["sentence"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        return dataset  


    def preprocess_dataset(self, 
                           glue_dataset: str = "sst2", 
                           truncate: int = None
                           ) -> Tuple[DataLoader, DataLoader]:
        """
        Preprocesses the dataset for SFL: tokenization, truncation, DataLoader
        """

        # Load dataset
        dataset = load_dataset("glue", glue_dataset)
        train_dataset, valid_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

        # Truncate dataset
        if truncate is not None:
            train_dataset = train_dataset.select(range(truncate))
            valid_dataset = valid_dataset.select(range(truncate))
            test_dataset = test_dataset.select(range(truncate))
        else:
            pass

        # Tokenize dataset
        train_dataset = self.tokenize(train_dataset)
        valid_dataset = self.tokenize(valid_dataset)
        test_dataset = self.tokenize(test_dataset)

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader


    def split_data(self, 
                   data: DataLoader, 
                   clients: int = 2) -> List[DataLoader]:
        """
        Splits the training and validation datasets into equally sized client portions.
        """
        dataset = data.dataset
        split_size = len(dataset) // clients
        remainder = len(dataset) % clients

        split_lengths = [split_size + 1 if i < remainder else split_size for i in range(clients)]
        splits = random_split(dataset, split_lengths)

        split_dataloaders = [DataLoader(split, batch_size=data.batch_size) for split in splits]

        return split_dataloaders


    def save_plots(self,
                   loss: List[float], 
                   accuracy: List[float],
                   title: str,
                   path= str) -> None:
        """
        Saves the loss and accuracy plots.
        """
        plt.figure(figsize=(10, 5))

        epochs = range(1, len(loss) + 1)

        # Plot the loss values
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'r', label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot the accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
        plt.title('Training vs. Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Title
        plt.suptitle(title)

        # Save the plot
        plt.savefig(path)


    def save_metric(self, metric: List[float], path: str) -> None:
        """
        Saves the metric (e.g. loss or accuracy) to the specified path.
        """
        np.array(metric)
        np.save(path, metric)


    def load_metric(self, path: str) -> List[float]:
        """
        Loads the metric (e.g. loss or accuracy) from the specified path.
        """
        metric = np.load(path)
        return metric