import pytorch_lightning

from models.bert_module import CustomBertModel

def main():
    model = CustomBertModel.from_pretrained("bert-base-uncased", num_labels=2)


if __name__ == "__main__":
    main()
