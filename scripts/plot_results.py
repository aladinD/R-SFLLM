import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.plotting import accumulate_client_metrics, accumulate_master_metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml


def fetch_relevant_directories(main_dir: str, dataset: str, model: str) -> list:
    """
    Traverse the main directory and fetch subdirectories that match the dataset and model criteria.
    
    Parameters:
    - main_dir (str): Path to the main directory.
    - dataset (str): Target dataset.
    - model (str): Target model.

    Returns:
    - list: List of subdirectories that match the criteria.
    """
    matched_dirs = []
    
    # List all subdirectories in the main directory
    subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    
    for subdir in subdirs:

        tags_path = os.path.join(main_dir, subdir, "tags.log")

        if os.path.exists(tags_path):
            with open(tags_path, 'r') as file:
                content = file.read()

                # Extracting tags from the content
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                if dataset in tags[0] and model in tags[1]:
                    matched_dirs.append(os.path.join(main_dir, subdir))
    
    return matched_dirs


def fetch_configuration(directory: str) -> dict:
    """
    Fetch and interpret the configuration from `config_tree.yaml` in the given directory.
    
    Parameters:
    - directory (str): Path to the target directory.

    Returns:
    - dict: Configuration dictionary.
    """
    config_path = os.path.join(directory, "config_tree.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def accumulate_client_metrics(config: dict, client_name: str, logs_path: str) -> None:
    """
    Reads and accumulates metrics for a specified client from all rounds.
    """
    # Extracting hyperparameters from the configuration
    task = config["task_name"]
    num_epochs = config["sfl"]["num_epochs"]
    num_rounds = config["sfl"]["num_rounds"]

    base_path = os.path.join(logs_path, "logs", "clients")
    client_dir = os.path.join(base_path, client_name)
    
    # List all rounds for the client
    rounds = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]
    rounds = rounds[:num_rounds]
    
    all_train_metrics = []
    all_val_metrics = []
    for idx, r in enumerate(rounds):
        metrics_path = os.path.join(client_dir, r, 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            
            # Update the epoch number by adding an offset
            df['epoch'] = df['epoch'] + idx * num_epochs
            
            # Split the metrics into training and validation
            if task == "sc":
                train_metrics = df[['epoch', 'train_loss', 'train_acc']]
                val_metrics = df[['epoch', 'val_loss', 'val_acc']]
            elif task == "ner":
                train_metrics = df[['epoch', 'train_loss', 'train_f1', 'train_precision', 'train_recall']]
                val_metrics = df[['epoch', 'val_loss', 'val_f1', 'val_precision', 'val_recall']]
            else:
                print("ERROR")
            
            all_train_metrics.append(train_metrics)
            all_val_metrics.append(val_metrics)
    
    # Concatenate metrics from all rounds
    train_df = pd.concat(all_train_metrics, ignore_index=True)
    val_df = pd.concat(all_val_metrics, ignore_index=True)
    
    # Drop rows where values are NaN
    if task == "sc":	
        train_df.dropna(subset=['train_acc'], inplace=True)        
        val_df.dropna(subset=['val_acc'], inplace=True)
    elif task == "ner":
        train_df.dropna(subset=['train_f1'], inplace=True)
        val_df.dropna(subset=['val_f1'], inplace=True)
    else:
        print("ERROR")

    # Reset index
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    return train_df, val_df


def accumulate_master_metrics(config: dict, logs_path: str) -> None:
    """
    Reads and accumulates metrics for the master model from all rounds.
    """
    # Extracting hyperparameters from the configuration
    num_epochs = config["sfl"]["num_epochs"]
    num_rounds = config["sfl"]["num_rounds"]

    base_path = os.path.join(logs_path, "logs", "master")
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'validation')
    
    # List all rounds for the master model
    rounds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    rounds = rounds[:num_rounds]
    
    all_train_metrics = []
    all_val_metrics = []
    for idx, r in enumerate(rounds):
        train_metrics_path = os.path.join(train_dir, r, 'metrics.csv')
        val_metrics_path = os.path.join(val_dir, r, 'metrics.csv')
        
        if os.path.exists(train_metrics_path):
            train_df = pd.read_csv(train_metrics_path)
            # Update the epoch number based on the round number
            train_df['epoch'] = train_df['epoch'] + idx * num_epochs - 1
            all_train_metrics.append(train_df)
        
        if os.path.exists(val_metrics_path):
            val_df = pd.read_csv(val_metrics_path)
            # Update the epoch number based on the round number
            val_df['epoch'] = val_df['epoch'] + idx * num_epochs - 1
            all_val_metrics.append(val_df)
    
    # Concatenate metrics from all rounds
    accumulated_train_df = pd.concat(all_train_metrics, ignore_index=True)
    accumulated_val_df = pd.concat(all_val_metrics, ignore_index=True)

    # Adjust master epoch numbering
    accumulated_train_df['epoch'] = accumulated_train_df['epoch'] + num_epochs
    accumulated_val_df['epoch'] = accumulated_val_df['epoch'] + num_epochs
    
    return accumulated_train_df, accumulated_val_df





# def generate_multiplot(main_dir: str, dataset: str, model: str, client_name="client_0"):
#     """
#     Generate a multiplot for all the subfolders that match the given dataset and model.
#     """
#     # Fetch all relevant directories based on the dataset and model
#     directories = fetch_relevant_directories(main_dir, dataset, model)
    
#     # Number of directories fetched
#     num_dirs = len(directories)
    
#     # Calculate the layout for the subplots
#     cols = 2
#     rows = (num_dirs + 1) // cols
    
#     # Create the main figure for multiplot
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
#     for idx, directory in enumerate(directories):
#         # Fetch the configuration for the current directory
#         config = fetch_configuration(directory)
        
#         # Determine the current subplot axis
#         ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
        
#         # Plot the metrics for the current directory on the current subplot axis
#         # client_train_df, client_val_df = accumulate_client_metrics(config, 'client_0_logger', directory)
#         client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", directory)
#         master_train_df, master_val_df = accumulate_master_metrics(config, directory)
        
#         # Extracting parameters from the configuration
#         task = config["task_name"]
#         if task == "sc":
#             # Plot val metrics for client and master model for sequence classification
#             # ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'client_0 per epoch val accuracies', color='blue', linestyle='-', marker='o')
#             ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{client_name} per epoch val accuracies', color='blue', linestyle='-', marker='o')
#             ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label='global val accuracy after each round', color='red', linestyle='-', marker='o')
#             ax.set_ylabel('Accuracy')
        
#         elif task == "ner":
#             # Plot F1 score metrics for client and master model for NER
#             # ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'client_0 per epoch val F1', color='blue', linestyle='-', marker='o')
#             ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{client_name} per epoch val F1', color='blue', linestyle='-', marker='o')
#             ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label='global val F1 after each round', color='red', linestyle='-', marker='o')
#             ax.set_ylabel('F1 Score')
        
#         main_title = f"Results for {config['model']['config']['pretrained_model_name_or_path']}"
#         ax.set_title(main_title)
#         ax.set_xlabel('Cumulative Epochs')
#         ax.legend()
#         ax.grid(True, alpha=0.5)
    
#     # Remove any unused subplots
#     for j in range(idx + 1, rows * cols):
#         fig.delaxes(axes.flatten()[j])


#     # Tight layout and save the multiplot
#     plt.tight_layout()
#     plt.savefig(os.path.join(main_dir, 'multiplot.png'))

def generate_multiplot(main_dir: str, dataset: str, model: str, client_name="client_0", plot_name="multiplot.png"):
    """
    Generate a multiplot for all the subfolders that match the given dataset and model.
    """
    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)
    
    # Number of directories fetched
    num_dirs = len(directories)
    
    # Calculate the layout for the subplots
    cols = 2
    rows = (num_dirs + 1) // cols
    
    # Create the main figure for multiplot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    for idx, directory in enumerate(directories):
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Fetch the scenario from the config or tags.log
        scenario = config["tags"][-1] if "tags" in config else None
        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                scenario = tags[-1]

        # Determine the current subplot axis
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
        
        # Plot the metrics for the current directory on the current subplot axis
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", directory)
        master_train_df, master_val_df = accumulate_master_metrics(config, directory)
        
        # Extracting parameters from the configuration
        task = config["task_name"]
        if task == "sc":
            # Plot val metrics for client and master model for sequence classification
            ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{client_name} per epoch val accuracies', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label='global val accuracy after each round', color='red', linestyle='-', marker='o')
            ax.set_ylabel('Accuracy')
        
        elif task == "ner":
            # Plot F1 score metrics for client and master model for NER
            ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{client_name} per epoch val F1', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label='global val F1 after each round', color='red', linestyle='-', marker='o')
            ax.set_ylabel('F1 Score')
        
        # Set scenario
        if scenario == "no_jammer":
            ax.set_title("No Jamming, only Wireless Impairments")
        elif scenario == "no_protection":
            ax.set_title("Adversarial Jamming without Protection")
        elif scenario == "w_protection":
            ax.set_title("Adversarial Jamming with Protection")
        else:
            ax.set_title("Baseline Performance without the Wireless Channel")

        ax.set_xlabel('Cumulative Epochs')
        ax.grid(True, alpha=0.5)
    
    # Remove any unused subplots
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    # Add a single legend for the whole figure
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=2)

    # Add a title for the whole multiplot
    fig.suptitle('Accuracies across Global Rounds and Epochs', fontsize=20)

    # Tight layout and save the multiplot
    plt.tight_layout()

    # Adjust spacing and layout
    plt.subplots_adjust(bottom=0.1)

    plt.savefig(os.path.join(main_dir, plot_name))





if __name__ == "__main__":
    generate_multiplot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42', 
                       dataset='sst2', 
                       model='roberta_for_sequence_classification',
                       plot_name="multiplot_sst2_roberta.png")
    


