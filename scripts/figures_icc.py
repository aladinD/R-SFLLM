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
import numpy as np

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


def training_curves(main_dir: str,
                    base_model: str,
                    sc_datasets: List[str],
                    ner_datasets: List[str],
                    plot_data: str = "client",
                    client_name: str = "client_0",
                    save_dir: str = ".",
                    plot_name: str = "jointmultiplot.png"
    ):
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

    subdirs: List[str] = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
        
    # Function to handle plotting for both SC and NER datasets
    def fetch_metrics_for_dataset(dataset: str, task: str):
        model = model_mapping[base_model][task]
        metric_name = 'acc' if task == 'sc' else 'f1'
        
        scenarios = []
        metrics = []
        xaxes = []
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
                    raise ValueError(f"Error fetching scenario for directory {directory}: {e}")

                # Fetch metrics for the current directory
                _, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
                _, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

                if plot_data in ["client", "both"]:
                    xaxis, metric = client_val_df["epoch"], client_val_df[f"val_{metric_name}"]
                if plot_data in ["master", "both"]:
                    xaxis, metric = master_val_df["epoch"], master_val_df[f"test_{metric_name}"]

                scenarios.append(scenario)
                xaxes.append(xaxis.to_numpy())
                metrics.append(metric.to_numpy())
        return scenarios, xaxes, metrics

    # Process SC datasets
    sc_metrics = []
    for dataset in sc_datasets:
        scenarios, xaxes, metrics = fetch_metrics_for_dataset(dataset, 'sc')
        print(scenarios)
        d = {s: (x, m) for s, x, m in zip(scenarios, xaxes, metrics)}
        sc_metrics.append(d)
    
    ner_metrics = []
    # Process NER datasets
    for dataset in ner_datasets:
        scenarios, xaxes, metrics = fetch_metrics_for_dataset(dataset, 'ner')
        print(scenarios)
        d = {s: (x, m) for s, x, m in zip(scenarios, xaxes, metrics)}
        ner_metrics.append(d)
    
    f, ax = plt.subplots(2, 2, figsize=(20, 14))
    ax = ax.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(scenarios)]
    colormap = {k: v for k, v in zip(scenarios, colors)}
    ylims = [[0.4, 1.], [0.5, 1.], [0., .5], [0., 1.]]
    ylabels = ["Accuracy"]*2 + ["F1 Score"]*2

    datasets_all = sc_datasets + ner_datasets
    metrics_all = sc_metrics + ner_metrics
    for idx, (tmp, a) in enumerate(zip(metrics_all, ax)):
        for k, v in tmp.items():
            a.plot(v[0], v[1], linestyle='-', marker='o', color=colormap[k])
            a.set_ylim(*ylims[idx])
            a.set_xlabel(r'Cumulative Epochs', fontsize=14)
            a.set_ylabel(ylabels[idx], fontsize=14)
            a.set_title(f'{datasets_all[idx].upper()}', fontsize=16)
            a.grid(True, alpha=0.5)
    
    f.legend(
        [label_mapping[scen] for scen in scenarios],
        loc='lower center', 
        ncol=len(scenarios), 
        fancybox=True, 
        shadow=True,
        fontsize=14
        )
    
    f.savefig(fname=os.path.join(save_dir, plot_name))
    
if __name__ == "__main__":
    training_curves(main_dir="/home/shared/plotting/", 
                    base_model='bert',
                    sc_datasets=['sst2', 'mrpc'],
                    ner_datasets=['wnut_17', 'conll2003'],
                    plot_data='client',
                    client_name="client_0",
                    save_dir = "./",
                    plot_name="joint_multiplot_bert.png")