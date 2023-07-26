from bert_module import CustomBertModel
from pytorch_lightning import LightningModule
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from typing import Dict


class Client(LightningModule):
    """
    LitModule Client class SFL.
    """
    def __init__(self,
                 name: str,
                 model: CustomBertModel,
                 train_data: DataLoader,
                 val_data: DataLoader) -> None: 
        super().__init__()
        self.name = name
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    
    def update_model(self, updates: Dict[str, Parameter]) -> None:
        """
        Updates the client model with the aggregated weight updates.
        """
        for gradient_name in updates.keys():
            for name, param in self.model.named_parameters():
                if name == gradient_name:
                    param.data = updates[gradient_name]