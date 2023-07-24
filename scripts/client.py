from manager import Manager
import torch
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from typing import Dict, List, Tuple

    
class Client:
    """
    Client class for SFL.
    """
    def __init__(self, 
                 name: str, 
                 model: BertForSequenceClassification,
                 manager: Manager, 
                 train_data: DataLoader,
                 valid_data: DataLoader) -> None:
        self.name = name
        self.model = model
        self.manager = manager
        self.train_data = train_data
        self.valid_data = valid_data

    
    def save_model(self, path: str) -> None:
        """
        Saves the current model to the specified path.
        """
        torch.save(self.model.state_dict(), path)


    def train_model(self, 
                    device: torch.device, 
                    checkpoint_path: str,
                    num_epochs: int = 1, 
                    learning_rate: float = 1e-5,
                    epsilon: float = 1e-6,
                    num_warmup_steps: float = 1256) -> Tuple[List[float], List[float]]:
        """
        Trains the client model on its data.
        """
        self.model.to(device)

        total_steps = len(self.train_data) * num_epochs

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

        # Training Loop
        global_loss = []
        global_accuracy = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            self.model.train()
            
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            # Training
            for step, batch in enumerate(tqdm(self.train_data)):

                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.detach().float()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Validation
            self.model.eval()

            for step, batch in enumerate(tqdm(self.valid_data)):

                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }

                with torch.no_grad():
                    outputs = self.model(**inputs)

                eval_loss = outputs.loss
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=1)
                true_labels = batch[2]

                total_correct += (predictions == true_labels).sum().item()
                total_samples += len(true_labels)
                total_loss += eval_loss.item()

            # Compute average loss and accuracy
            average_loss = total_loss / len(self.train_data)
            accuracy = total_correct / total_samples

            print(f"Average Loss: {average_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

            global_loss.append(average_loss.cpu())
            global_accuracy.append(accuracy)


        # Save model
        # checkpoint_path = f"/home/munichcenter/code/bert/sst2/fedavg/results/bert_results/checkpoints/{self.name}_model.pt"
        checkpoint_path = checkpoint_path + f"/{self.name}_model.pt"
        self.save_model(path=checkpoint_path)

        print("Training complete!")

        return global_loss, global_accuracy


    def evaluate_model(self, 
                       model_path: str, 
                       device: torch.device, 
                       valid_data: DataLoader) -> None:
        """
        Evaluates the client model on its validation data.
        """
        print("START EVALUATION")
        self.valid_data = valid_data
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(tqdm(self.valid_data)):

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            eval_loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            true_labels = batch[2]

            total_correct += (predictions == true_labels).sum().item()
            total_samples += len(true_labels)
            total_loss += eval_loss.item()

        # Compute average loss and accuracy
        average_loss = total_loss / len(self.valid_data)
        accuracy = total_correct / total_samples

        print(f"Average Loss: {average_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")


    def evaluate_global_model(self, 
                              device: torch.device, 
                              valid_data: DataLoader) -> Tuple[float, float]:
        """
        Evaluates a global model (e.g. fedavg-ed) on its validation data and 
        returns the average loss and accuracy.
        """
        print("START EVALUATION")
        self.valid_data = valid_data
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(tqdm(self.valid_data)):

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            eval_loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            true_labels = batch[2]

            total_correct += (predictions == true_labels).sum().item()
            total_samples += len(true_labels)
            total_loss += eval_loss.item()

        # Compute average loss and accuracy
        average_loss = total_loss / len(self.valid_data)
        accuracy = total_correct / total_samples

        print(f"Average Loss: {average_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        return average_loss, accuracy


    def update_model(self, updates: Dict[str, Parameter]) -> None:
        """
        Updates the client model with the aggregated weight updates.
        """
        for gradient_name in updates.keys():
            for name, param in self.model.named_parameters():
                if name == gradient_name:
                    param.data = updates[gradient_name]