import torch
from transformers import BertModel, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder
from typing import Tuple, Union


class CustomBertEncoder(BertEncoder):
    """
    Custom BERT Encoder class for the attention layer. 
    """
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def forward(self, 
                inputs_embeds, 
                attention_mask=None, 
                head_mask=None) -> BaseModelOutputWithPoolingAndCrossAttentions:

        hidden_states = inputs_embeds
        
        # Ensure attention mask dtype is the same as hidden_states.dtype and
        # modify attention mask values for masked positions
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        for layer in self.layer:
            outputs = layer(hidden_states, attention_mask, head_mask)
            hidden_states = outputs[0]
    
        return hidden_states


class CustomBertModel(BertForSequenceClassification):
    """
    Custom BERT Model class for testing SFL.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Create separate modules for embedding, attention, and head layers
        self.embedding = BertModel(config)
        self.attention = CustomBertEncoder(config)
        self.head = self.classifier

    
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