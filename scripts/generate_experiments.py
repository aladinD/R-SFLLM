"""
Generates experiment configs for all tasks
"""
from omegaconf import OmegaConf
import itertools
import copy
import os
import rootutils

def main(experiment_dir: str = os.path.split(__file__)[0]) -> None:
    all_dict = {
        "ner": {
            "datamodule": ["conll2003.yaml", "conll2012_ontonotesv5.yaml", "wnut_17.yaml"],
            "model": ["bert_for_token_classification.yaml", "roberta_for_token_classification.yaml"],
            "wireless": [None, "no_jammer.yaml", "no_protection.yaml", "w_protection.yaml"]
        }, 
        "sc": {
            "datamodule": ["sst2.yaml", "mrpc.yaml", "qnli.yaml"],
            "model": ["bert_for_sequence_classification.yaml", "roberta_for_sequence_classification.yaml"],
            "wireless": [None, "no_jammer.yaml", "no_protection.yaml", "w_protection.yaml"]
        }
    }
    config_scaffold = {
        "defaults": [{f"override /{o}": None} for o in all_dict["ner"].keys()],
        "tags": [],
        "task_name": None
    }
    print(config_scaffold)
    for task, vals in all_dict.items():
        task_dir = os.path.join(experiment_dir, task)
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)
        
        dmodules = vals["datamodule"]
        models = vals["model"]
        wireless = vals["wireless"]
        names = list(vals.keys())
        combinations = itertools.product(dmodules, models, wireless)
        for comb in combinations:
            conf_dict = copy.deepcopy(config_scaffold)
            conf_dict["task_name"] = task
            tags = list(map(lambda x: x.split(".")[0] if isinstance(x, str) else str(x), comb))
            conf_dict["tags"] = tags
            for i, c in enumerate(comb):
                conf_dict["defaults"][i][f"override /{names[i]}"] = c
            conf = OmegaConf.create(conf_dict)
            tags_proc = tags
            tags_proc[1] = tags_proc[1].split("_")[0]
            tags_proc[2] = "baseline" if tags_proc[2] == "None" else tags_proc[2]
            exp_name = "_".join(tags_proc)
            with open(os.path.join(task_dir, f"{exp_name}.yaml"), "w") as f:
                f.write("# @package _global_ \n")
                OmegaConf.save(config=conf, f=f)
            print(f"Generating experiment: {exp_name}")


if __name__ == "__main__":
    project_root = rootutils.find_root(".", ".project-root")
    print(project_root)
    exp_dir = os.path.join(project_root, "configs", "experiment")
    main(experiment_dir=exp_dir)