import argparse
from german_data import loader
from model_trainers.byt5_pretrained import train_transformer, eval_transformer
from inseq_analysis import do_contrast_analysis, do_analysis
import os
import os.path

ACTIONS = [
    "train_transformer",
    "eval_transformer",
    "do_analysis",
    "do_contrast_analysis"
]

parser = argparse.ArgumentParser(prog="Neural model of morphological inflection",
                                 description= """This program will apply state-of-the-art interpretability tools to 
                                                 dissect a neural model of morphological inflection and investigate whether its 
                                                 strategies reflect (or not) established linguistic principles.""")

parser.add_argument("--action", type=str,
                    choices=ACTIONS, help="The specific module to run", required=True)

parser.add_argument("--local_load", type=str, default="yes",
                    help="Redownload source data or load from local dataset")

parser.add_argument("--epochs", type=int, default=3,
                    help="The number of epochs the model will be trained")

parser.add_argument("--name", type=str, default="byt5finetune_ruhi",
                    help="The name to use for the model")


if __name__ == "__main__":

    options = parser.parse_args()
    cwd = os.getcwd()

    dataset_filepath = os.path.join(cwd, "german_data")
    model_filepath = os.path.join(cwd, "models")

    # load dataset
    if options.action == "train_transformer" or options.action == "eval_transformer":        
        ds = None
        if options.local_load.lower() == "yes":
            ds = loader.load_data(dataset_filepath, True)
        else:
            ds = loader.load_data(dataset_filepath, False)
        if ds == None:
            print("#### No dataset found ####")

    # run specific modules
    if options.action == "train_transformer":
        train_transformer(ds, options.name, model_filepath, options.epochs)

    elif options.action == "eval_transformer":
        eval_transformer(ds, model_filepath, options.name)

    elif options.action == "do_analysis":
        model_f = os.path.join(model_filepath, options.name)
        input_text = input("> Enter your input text in format morphosyntactic tags: noun\n")
        do_analysis(model_f, input_text)

    elif options.action == "do_contrast_analysis":
        model_f = os.path.join(model_filepath, options.name)
        input_text = input("> Enter your input text in format morphosyntactic tags: noun\n")
        truth = input("> Enter the original target")
        contrast = input("> Enter contrast target")
        do_contrast_analysis(model_f, input_text, contrast, truth)
    # TODO: challenge part
    else:
        print("no valid action selected!")
