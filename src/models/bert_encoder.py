from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder

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