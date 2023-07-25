from pytorch_lightning import LightningModule
import torch
from torchmetrics.classification.accuracy import Accuracy
from transformers import BertForSequenceClassification, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Any, List, Tuple, Union
from .bert_encoder import CustomBertEncoder


class CustomBertModel(BertForSequenceClassification, LightningModule):
    """
    LitModule BERT Model class for testing SFL.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Create separate modules for embedding, attention, and head layers
        self.embedding = BertModel(config)
        self.attention = CustomBertEncoder(config)
        self.head = self.classifier

        # Assign performance metrics
        self.accuracy = Accuracy(task="binary", num_classes=2)  # Add to config

    
    def forward(self, 
                input_ids, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None, 
                labels=None) -> Tuple[Union[SequenceClassifierOutput, Tuple[torch.tensor]], torch.tensor]:

        # Embedding layer pass
        embeddings = self.embedding(input_ids=input_ids, 
                                    attention_mask=attention_mask, 
                                    token_type_ids=token_type_ids, 
                                    position_ids=position_ids)[0]
        
        # Attention layer pass
        attentions = self.attention(inputs_embeds=embeddings, 
                                    attention_mask=attention_mask,
                                    head_mask=head_mask)
   
        # Head layer pass
        logits = self.head(attentions[:, 0, :])
        
        # Compute loss if labels are provided
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return logits
        

    def training_step(self, batch: Any) -> torch.tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }

        outputs = self.forward(**inputs)
        loss = outputs.loss
        accuracy = self.accuracy(torch.argmax(outputs.logits, dim=1), inputs["labels"])
        
        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def validation_step(self, batch: Any) -> torch.tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        loss = outputs.loss
        accuracy = self.accuracy(torch.argmax(outputs.logits, dim=1), inputs["labels"])

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def predict_step(self, batch: Any) -> torch.tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        return preds


    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, eps=1e-6)   # Add to congig! 
        scheduler = torch.optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1256)  # Add to congig! 
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }