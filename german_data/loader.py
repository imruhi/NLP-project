from datasets import load_dataset
import os


def load_data():
    dataset = load_dataset("csv", data_files={"train": ["deu_100.tsv", "deu_200.tsv", "deu_300.tsv", 
                                                        "deu_400.tsv", "deu_500.tsv", "deu_600.tsv"],
                                              "test": "deu_gold.tsv", 
                                              "validation": "deu_dev.tsv"})

    # 
    return dataset

if __name__ == "__main__":
    data = load_data()
    for split in data.keys():
        print(data[split][0])