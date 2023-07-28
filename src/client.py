from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from models.bert_module import CustomBertModelModule


class Client:
    """
    Client class for SFL.
    """
    def __init__(self, 
                 name: str, 
                 model: CustomBertModelModule,
                 train_data: DataLoader,
                 val_data: DataLoader) -> None:
        self.name = name
        self.model = model
        self.train_data = train_data
        self.val_data = valid_data

    
    def save_model(self, path: str) -> None:
        """
        Saves the current model to the specified path.
        """
        torch.save(self.model.state_dict(), path)


    def update_model(self, updates: Dict[str, Parameter]) -> None:
        """
        Updates the client model with the aggregated weight updates.
        """
        for gradient_name in updates.keys():
            for name, param in self.model.named_parameters():
                if name == gradient_name:
                    param.data = updates[gradient_name]