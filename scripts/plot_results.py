import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from src.utils.plotting import accumulate_client_metrics, accumulate_master_metrics
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
        
        # Set the y-axis scale
        ax.set_ylim(0.5, 1)

        # Set scenario titles
        title_mapping = {
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Jamming without Protection",
            "w_protection": "Adversarial Jamming with Protection",
        }
        ax.set_title(title_mapping.get(scenario, "Baseline Performance without the Wireless Channel"))
        
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


def generate_joint_plot(main_dir: str, 
                        dataset: str, 
                        model: str, 
                        plot_data: str = "client", 
                        client_name="client_0", 
                        plot_name="jointplot.png"):
    """
    Generate a joint plot for all subfolders that match the given dataset and model.
    
    Parameters:
    - main_dir: Main directory path.
    - dataset: Dataset name.
    - model: Model name.
    - plot_data: A string which can be "client", "master", or "both" to decide which data to plot.
    - client_name: The name of the client.
    - plot_name: The name of the saved plot.
    
    """
    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)
    
    # Create the main figure for the joint plot
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Loop through each directory and plot the data
    for directory in directories:
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle potential errors while fetching scenario
        try:
            scenario = config["tags"][-1] if "tags" in config else None
            if not scenario:
                with open(os.path.join(directory, "tags.log"), 'r') as file:
                    content = file.read()
                    tags = [tag.strip() for tag in content.strip("[]").split(",")]
                    scenario = tags[-1]
        except Exception as e:
            print(f"Error fetching scenario for directory {directory}: {e}")
            continue
        
        # Fetch metrics for the current directory
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", directory)
        master_train_df, master_val_df = accumulate_master_metrics(config, directory)
        
        # Extracting task from the configuration
        task = config.get("task_name", "")
        
        # Plot client data based on the task
        label_mapping = {
            "None": "Baseline Performance without the Wireless Channel",
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Jamming without Protection",
            "w_protection": "Adversarial Jamming with Protection",
        }
        label_scenario = label_mapping.get(scenario, scenario)

        if plot_data in ["client", "both"]:
            if task == "sc":
                ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{label_scenario}', linestyle='-', marker='o')
            elif task == "ner":
                ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{label_scenario}', linestyle='-', marker='o')
        
        # Plot master data based on the task
        if plot_data in ["master", "both"]:
            if task == "sc":
                ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label=f'{label_scenario}', linestyle='--', marker='x')
            elif task == "ner":
                ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label=f'{label_scenario}', linestyle='--', marker='x')

    # Set the axis parameters
    ax.set_ylim(0.4, 1)
    ax.set_xlabel('Cumulative Epochs')

    # Set title
    main_title_fontsize = 16
    subtitle_fontsize = 10
    model = config["tags"][1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config["tags"][0]
    num_labels = config["model"]["config"]["num_labels"]
    num_clients = config["sfl"]["num_clients"]
    num_epochs = config["sfl"]["num_epochs"]
    num_rounds = config["sfl"]["num_rounds"]

    if task == "sc":
        ax.set_ylabel('Accuracy')
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        ax.set_ylabel('F1 Score')
        main_title = 'F1 Scores across Global Rounds and Epochs'

    subtitle = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}"
    plt.suptitle(main_title, fontsize=main_title_fontsize)  
    plt.title(subtitle, fontsize=subtitle_fontsize) 

    ax.grid(True, alpha=0.5)
    # ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4)
    plt.subplots_adjust(bottom=0.25)


    # Adjust spacing and layout
    plt.tight_layout()

    plt.savefig(os.path.join(main_dir, plot_name))


def generate_joint_bar_plot(main_dir: str, 
                            dataset: str, 
                            model: str, 
                            plot_data: str = "client", 
                            client_name="client_0", 
                            plot_name="barplot.png"):
    """
    Generate a joint bar plot for all subfolders that match the given dataset and model.
    """
    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)
    
    # Create the main figure for the bar plot
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Lists to store bar plot data
    scenarios = []
    client_metrics = []
    master_metrics = []
    
    # Loop through each directory and accumulate data
    for directory in directories:
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle potential errors while fetching scenario
        try:
            scenario = config["tags"][-1] if "tags" in config else None
            if not scenario:
                with open(os.path.join(directory, "tags.log"), 'r') as file:
                    content = file.read()
                    tags = [tag.strip() for tag in content.strip("[]").split(",")]
                    scenario = tags[-1]
        except Exception as e:
            print(f"Error fetching scenario for directory {directory}: {e}")
            continue
        
        # Fetch metrics for the current directory
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", directory)
        master_train_df, master_val_df = accumulate_master_metrics(config, directory)
        
        # Extracting task from the configuration
        task = config.get("task_name", "")
        
        # Plot client data based on the task
        label_mapping = {
            "None": "Baseline Performance without the Wireless Channel",
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Jamming without Protection",
            "w_protection": "Adversarial Jamming with Protection",
        }

        # label_scenario = label_mapping.get(scenario, scenario)
        label_scenario = label_mapping.get(scenario, scenario).replace(", ", ",\n")

        scenarios.append(label_scenario)
        
        # Accumulate metrics for bar plotting
        if task == "sc":
            client_metrics.append(client_val_df['val_acc'].iloc[-1])  # Assuming you want the last epoch's data
            master_metrics.append(master_val_df['test_acc'].iloc[-1])
        elif task == "ner":
            client_metrics.append(client_val_df['val_f1'].iloc[-1])
            master_metrics.append(master_val_df['test_f1'].iloc[-1])

    # Plotting bars
    bar_width = 0.35
    indices = list(range(len(scenarios)))
    
    if plot_data in ["client", "both"]:
        ax.bar(indices, client_metrics, bar_width, label='Client Model', alpha=0.8)
        
    if plot_data in ["master", "both"]:
        # If both client and master data are plotted, shift master bars to the right for clarity
        if plot_data == "both":
            indices = [i + bar_width for i in indices]
        ax.bar(indices, master_metrics, bar_width, label='Global SFL Model', alpha=0.8)
    
    # Formatting
    ax.set_ylim(0.4, 1)
    ax.set_xlabel('Scenarios')
    ax.set_ylabel('Metric Value')
    ax.set_xticks([i + bar_width/2 for i in range(len(scenarios))])
    # ax.set_xticklabels(scenarios, rotation=30, ha='right')
    ax.set_xticklabels(scenarios, ha='center')

    # Set title
    main_title_fontsize = 16
    subtitle_fontsize = 10
    model = config["tags"][1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config["tags"][0]
    num_labels = config["model"]["config"]["num_labels"]
    num_clients = config["sfl"]["num_clients"]
    num_epochs = config["sfl"]["num_epochs"]
    num_rounds = config["sfl"]["num_rounds"]

    if task == "sc":
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        main_title = 'F1 Scores across Global Rounds and Epochs'

    subtitle = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}"
    plt.suptitle(main_title, fontsize=main_title_fontsize)  
    plt.title(subtitle, fontsize=subtitle_fontsize)

    ax.grid(True, alpha=0.5)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4)
    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=2)
    plt.subplots_adjust(bottom=0.25)

    # Adjust spacing and layout
    plt.tight_layout()

    plt.savefig(os.path.join(main_dir, plot_name))



if __name__ == "__main__":
    generate_multiplot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42', 
                       dataset='sst2', 
                       model='roberta_for_sequence_classification',
                       client_name="client_0",
                       plot_name="multiplot_sst2_roberta.png")
    
    generate_joint_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42', 
                       dataset='sst2', 
                       plot_data='client',
                       model='bert_for_sequence_classification',
                       client_name="client_0",
                       plot_name="jointplot_sst2_bert.png")
    
    generate_joint_bar_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42',
                            dataset='sst2',
                            model='roberta_for_sequence_classification',
                            plot_data='both',
                            client_name="client_0",
                            plot_name="barplot_sst2_roberta.png")
    

