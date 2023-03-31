# Natural Language Processing Final Project
Neural Model(s) of Morphological Inflection

This repository is part of a course project for the course Natural Language Processing of the [University of Groningen](https://www.rug.nl/).

To create a new virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

All dependecies needed to run can be installed by running the following command:

```bash
pip3 install -r requirements.txt
```

Run the code using the following command in bash: 
```bash
python3 main.py --action eval_transformer --local_load no
```

## Running the code

All experiments can be started and controlled from the command line. Start main.py with the specified option by running the following command:

```bash
python3 main.py --action train_transformer
```

The available actions are:

- `train_transformer`: Finetune the Byt5 model
- `eval_transformer`: Evaluate an existing Byt5 model
- `do_analysis`: Compute attribution scores (using Inseq) for specific input
- `do_contrast_analysis`: Compute contrastive attribution scores (using Inseq) for specific input, original target and contrastive target 
- `get_average_score`: Extract statistics over the attributions of your entire test set

Additionally, the following options can be specified to further configure the program:

|Option|Description|Default Value|
|------|-----------|-------------|
|`local_load`|Load from local dataset or create a new one from source data|yes|
|`epochs`|The number of epochs to finetune the model for|5|
|`name`|The name used for the finetuned model|"byt5finetune_ruhi600"|
|`correct_preds`|Choose the correct preds while calculating average attribution scores or not|yes|
