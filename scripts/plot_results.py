import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.plotting import accumulate_client_metrics, accumulate_master_metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import List, Dict


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


def fetch_configuration(directory: str) -> DictConfig:
    """
    Fetch and interpret the configuration from `config_tree.yaml` in the given directory.
    
    Parameters:
    - directory (str): Path to the target directory.

    Returns:
    - DictConfig: Configuration dictionary.
    """
    config_path = os.path.join(directory, "config_tree.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return OmegaConf.create(config)


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

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Fetch the scenario from the config or tags.log
        scenario = config.tags[-1] if "tags" in config else None
        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                scenario = tags[-1]

        # Determine the current subplot axis
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]

        # Plot the metrics for the current directory on the current subplot axis
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        # Extracting parameters from the configuration
        task = config.task_name
        if task == "sc":
            # Plot val metrics for client and master model for sequence classification
            ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{client_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label='Global SFL Model', color='red', linestyle='-', marker='o')
            
            ax.set_ylabel('Accuracy')
        
        elif task == "ner":
            # Plot F1 score metrics for client and master model for NER            
            ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{client_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label='Global SFL Model', color='red', linestyle='-', marker='o')
            
            ax.set_ylabel('F1 Score')
        
        # Set the y-axis scale
        ax.set_ylim(0.4, 1)

        # Set scenario titles
        title_mapping = {
            "None": "Baseline Performance without the Wireless Channel",
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

    # Title
    main_title_fontsize = 20
    subtext_fontsize = 12
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

    if task == "sc":
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        main_title = 'F1 Scores across Global Rounds and Epochs'

    # fig.suptitle('Classification Accuracies across Global Rounds and Epochs', fontsize=20)
    subtext = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}" 
    
    fig.suptitle(main_title, fontsize=main_title_fontsize)
    fig.text(0.5, 0.92, subtext, ha='center', va='center', fontsize=subtext_fontsize)
    # fig.text(0.5, 0.92, 'Your Sub-Title Here', ha='center', va='center', fontsize=16)

    # Tight layout and save the multiplot
    plt.tight_layout()

    # Adjust spacing and layout
    # plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(bottom=0.1, top=0.85)

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

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Handle potential errors while fetching scenario
        try:
            scenario = config.tags[-1] if "tags" in config else None
            if not scenario:
                with open(os.path.join(directory, "tags.log"), 'r') as file:
                    content = file.read()
                    tags = [tag.strip() for tag in content.strip("[]").split(",")]
                    scenario = tags[-1]
        except Exception as e:
            print(f"Error fetching scenario for directory {directory}: {e}")
            continue
        
        # Fetch metrics for the current directory
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)
        
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
            client_metrics.append(client_val_df['val_acc'].iloc[-1]) 
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
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

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



def generate_joint_plot(main_dir: str, 
                        dataset: str, 
                        model: str, 
                        plot_data: str = "client", 
                        client_name="client_0", 
                        plot_name="jointplot.png",
                        ax=None):
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
    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax_provided = False

    # Loop through each directory and plot the data
    for directory in directories:
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Handle potential errors while fetching scenario
        try:
            scenario = config.tags[-1] if "tags" in config else None
            if not scenario:
                with open(os.path.join(directory, "tags.log"), 'r') as file:
                    content = file.read()
                    tags = [tag.strip() for tag in content.strip("[]").split(",")]
                    scenario = tags[-1]
        except Exception as e:
            print(f"Error fetching scenario for directory {directory}: {e}")
            continue
        
        # Fetch metrics for the current directory
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)
        
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
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

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

    # plt.savefig(os.path.join(main_dir, plot_name))
    if not ax_provided:
        plt.savefig(os.path.join(main_dir, plot_name))



def generate_joint_multiplot(main_dir: str,
                             base_model: str,
                             sc_datasets: List[str],
                             ner_datasets: List[str],
                             plot_data: str = "client",
                             client_name: str = "client_0",
                             plot_name: str = "jointmultiplot.png"):
    """
    Generate a joint multiplot for all subfolders that match the given datasets and model.
    
    Parameters:
    - main_dir: Main directory path.
    - base_model: Base model name (e.g., 'roberta' or 'bert').
    - sc_datasets: List of SC dataset names.
    - ner_datasets: List of NER dataset names.
    - plot_data: A string which can be "client", "master", or "both" to decide which data to plot.
    - client_name: The name of the client.
    - plot_name: The name of the saved plot.
    """
    
    # Model Mapping
    model_mapping: Dict[str, Dict[str, str]] = {
        'roberta': {
            'sc': 'roberta_for_sequence_classification',
            'ner': 'roberta_for_token_classification'
        },
        'bert': {
            'sc': 'bert_for_sequence_classification',
            'ner': 'bert_for_token_classification'
        }
    }
    
    # Label Mapping
    label_mapping: Dict[str, str] = {
        "None": "Baseline Performance without the Wireless Channel",
        "no_jammer": "No Jamming, only Wireless Impairments",
        "no_protection": "Adversarial Jamming without Protection",
        "w_protection": "Adversarial Jamming with Protection"
    }

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # Get the list of subdirectories in main_dir
    subdirs: List[str] = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    # Initialize an empty set to store unique legend labels
    legend_entries = {}

    # Function to handle plotting for both SC and NER datasets
    def plot_data_for_dataset(row: int, col: int, dataset: str, task: str):
        model = model_mapping[base_model][task]
        metric = 'acc' if task == 'sc' else 'f1'
        ylabel = r'Accuracy' if task == 'sc' else r'F1 Score'
        
        for subdir in subdirs:
            directories = fetch_relevant_directories(subdir, dataset, model)
            if not directories:
                continue

            for directory in directories:
                # Fetch the configuration for the current directory
                config = fetch_configuration(directory)

                # Handle relative paths
                relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
                relative_path = os.path.join(*relative_path_components[3:4])
                relative_path = os.path.join(relative_path, 'logs/')  
                config.paths.log_path = os.path.join(subdir, relative_path)

                # Get scenario
                try:
                    scenario = config.tags[-1] if "tags" in config else None
                    if not scenario:
                        with open(os.path.join(directory, "tags.log"), 'r') as file:
                            content = file.read()
                            tags = [tag.strip() for tag in content.strip("[]").split(",")]
                            scenario = tags[-1]
                except Exception as e:
                    print(f"Error fetching scenario for directory {directory}: {e}")
                    continue

                # Fetch metrics for the current directory
                client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
                master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

                # Plot data
                ax = axs[row][col]
                label_scenario = label_mapping.get(scenario, scenario)

                label = f'{label_scenario}'
                if plot_data in ["client", "both"]:
                    line1, = ax.plot(client_val_df['epoch'], client_val_df[f'val_{metric}'], label=f'{label_scenario}', linestyle='-', marker='o')
                    legend_entries[label] = line1
                
                if plot_data in ["master", "both"]:
                    line2, = ax.plot(master_val_df['epoch'], master_val_df[f'test_{metric}'], label=f'{label_scenario}', linestyle='--', marker='x')
                    legend_entries[label] = line2

                # Set the axis parameters for the subplot
                ax.set_ylim(0.4, 1) if task == 'sc' or dataset == "conll2003" else ax.set_ylim(0.05, 0.5)
                ax.set_xlabel(r'Cumulative Epochs', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                ax.set_title(f'{dataset.upper()}',  fontsize=16)
                ax.grid(True, alpha=0.5)

    # Process SC datasets
    for idx, dataset in enumerate(sc_datasets):
        plot_data_for_dataset(0, idx, dataset, 'sc')

    # Process NER datasets
    for idx, dataset in enumerate(ner_datasets):
        plot_data_for_dataset(1, idx, dataset, 'ner')

    fig.legend(handles=legend_entries.values(), 
               labels=legend_entries.keys(), 
               loc='lower center', 
               ncol=len(legend_entries), 
               fancybox=True, 
               shadow=True,
               fontsize=14)
    
    fig.suptitle(r'Performance Metrics for SC and NER NLP Tasks', fontsize=18, fontweight='bold')


    # Set global title, layout, and save figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.09)  
    plt.savefig(os.path.join(main_dir, plot_name))



if __name__ == "__main__":

    generate_joint_multiplot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/',
                            base_model='roberta',
                            sc_datasets=['sst2', 'mrpc'],
                            ner_datasets=['wnut_17', 'conll2003'],
                            plot_data='client',
                            client_name="client_0",
                            plot_name="jointmultiplot_roberta.png")
    
    # generate_multiplot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42', 
    #                    dataset='sst2', 
    #                    model='roberta_for_sequence_classification',
    #                    client_name="client_0",
    #                    plot_name="multiplot_sst2_roberta.png")
    
    # generate_joint_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42', 
    #                    dataset='sst2', 
    #                    plot_data='client',
    #                    model='roberta_for_sequence_classification',
    #                    client_name="client_0",
    #                    plot_name="jointplot_sst2_roberta.png")
    
    # generate_joint_bar_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/2023-10-02_22-55-42',
    #                         dataset='sst2',
    #                         model='roberta_for_sequence_classification',
    #                         plot_data='both',
    #                         client_name="client_0",
    #                         plot_name="barplot_sst2_roberta.png")
    




    # generate_multiplot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/icc_logs_ner_vlad', 
    #                    dataset='conll2003', 
    #                    model='roberta_for_token_classification',
    #                    client_name="client_0",
    #                    plot_name="multiplot_conll2003_roberta.png")
    
    # generate_joint_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/icc_logs_ner_vlad', 
    #                    dataset='conll2003', 
    #                    plot_data='client',
    #                    model='roberta_for_token_classification',
    #                    client_name="client_0",
    #                    plot_name="jointplot_conll2003_roberta.png")
    
    # generate_joint_bar_plot(main_dir='/home/aladin/refactoring/resilient_sfl/logs/sc/multiruns/icc_logs_ner_vlad',
    #                         dataset='conll2003',
    #                         model='roberta_for_token_classification',
    #                         plot_data='both',
    #                         client_name="client_0",
    #                         plot_name="barplot_conll2003_roberta.png")