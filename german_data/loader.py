from datasets import load_dataset, load_from_disk, load_metric
from transformers import ByT5Tokenizer
import torch
import os 
import os.path

MODEL_CHECKPOINT = 'google/byt5-base'

def load_data(file_path, load_from_local = True):
    """ Load data from .csv or previously saved hf dataset 
    :param file_path: str, absolute location of the path where data is located
    :param load_from_local: boolean, true if want to load saved dataset
    """
    dataset_filepath = os.path.join(file_path, "german_noun_pl.hf")

    if not load_from_local: 
        print("#### Creating and tokenizing dataset #### ")
        dataset = load_dataset("csv", data_files={  
                                                    "train": os.path.join(file_path, "deu_600.csv"),
                                                    "test": os.path.join(file_path, "deu_gold.csv"), 
                                                    "validation": os.path.join(file_path, "deu_dev.csv")    
                                                }
                                )

        print(f"#### Removed {dataset.cleanup_cache_files()} cached files")    
        
        # uncomment below 3 lines for local testing
        # dataset["train"] = dataset["train"].select(range(0,3))
        # dataset["test"] = dataset["test"].select(range(0,3))
        # dataset["validation"] = dataset["validation"].select(range(0,3))
        
        dataset = preprocess(dataset)
        dataset.save_to_disk(dataset_filepath)
    
    else:
        print("#### Loading dataset from local #### ")
        dataset = load_from_disk(dataset_filepath)
        # uncomment below 3 lines for local testing
        # dataset["train"] = dataset["train"].select(range(0,3))
        # dataset["test"] = dataset["test"].select(range(0,3))
        # dataset["validation"] = dataset["validation"].select(range(0,3))

    return dataset

def format_input(example):
    """ format the lemma and tag columns into {lemma, tag}
    tags are in UniMorph format
    """
    return {"inflection": {"input": example["lemma"], "tag": example["tag"],
            "output": example["target_form"]}}

# https://huggingface.co/docs/datasets/use_dataset
def preprocess(dataset):
    for split in dataset.keys():
        dataset[split] = dataset[split].map(format_input, load_from_cache_file=False)
        dataset[split] = dataset[split].remove_columns(["lemma", "tag", "target_form"])
        
    return dataset
    
if __name__ == "__main__":
    # test purpose
    load_from_local = False
    cwd = os.getcwd()
    data = load_data(cwd, load_from_local)

    for split in data.keys():
        print(split, data[split][0])
        print("------------------------------")
