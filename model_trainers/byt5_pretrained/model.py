from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch
import evaluate
import gc
import numpy as np
import os.path
import matplotlib.pyplot as plt

MODEL_CHECKPOINT = "google/byt5-base"

def preprocess_function(examples):
    """ To process the input before using it in model (tokenization)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    max_input_length = 128
    max_target_length = 128

    inputs = [ex["tag"] + ": " + ex["input"] for ex in examples["inflection"]]
    targets = [ex["output"] for ex in examples["inflection"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_transformer(dataset, model_name, model_dir, num_epochs = 3):
    """ Finetune the byT5 transformer model and save it at model_dir
    """
    output_dir_model = os.path.join(model_dir, model_name, "checkpoints")
    logging_dir_model = os.path.join(model_dir, model_name, "logging")
    model_save_dir = os.path.join(model_dir, model_name)

    # try out to remove cache
    gc.collect()
    torch.cuda.empty_cache() 
    
    # cuda works only if you have an nvidia gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"#### training on {device} ####")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print("#### Tokenizing dataset ####")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    model.to(device)

    n_beams = 3
    max_gen_len = 36 # longest german noun has 36 letters
    num_epochs = 3
    train_batch_size = 2 # change it to something lower if you get memory error
    eval_batch_size = 2 
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir_model,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit = 3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        metric_for_best_model="cer", # metric to reduce (Character Error Rate)
        generation_max_length=max_gen_len,
        optim="adafactor",
        generation_num_beams=n_beams, # for beam search in decoder
        logging_strategy="epoch",
        logging_dir=logging_dir_model
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("cer") # metric used by byt5 (character error rate)

    ########

    def postprocess_text(preds, labels):
        """ Format the output so that metrics can be computed
        """
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        """ Compute metrics based on CER
        """
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"cer": result}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    ########
    idx_end = 100

    while idx_end != 700:
        model_save_dir = os.path.join(model_dir, model_name + idx_end)
        trainer = Seq2SeqTrainer(
                                    model,
                                    args,
                                    train_dataset=tokenized_datasets["train"][0:idx_end],
                                    eval_dataset=tokenized_datasets["validation"],
                                    data_collator=data_collator,
                                    tokenizer=tokenizer,
                                    compute_metrics=compute_metrics
                                )

        print("#### Training model ####")
        train_result = trainer.train()

        trainer.save_model(output_dir=model_save_dir)

        # try out to remove cache
        gc.collect()
        torch.cuda.empty_cache() 


def eval_transformer(dataset, model_f, data_f, model_name):
    """ Get direct accuracy and CER on test set
    """
    model_dir = os.path.join(model_f, model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print("#### Tokenizing dataset ####")
    dataset = dataset["test"]["inflection"]
    input_model = []
    labels = []

    for i in range(len(dataset)):
        input_model.append(dataset[i]["tag"] + ": " + dataset[i]["input"])
        labels.append(dataset[i]["output"])

    model_inputs = tokenizer(input_model, padding="max_length", max_length=36, return_tensors="pt")
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    print("#### Generating outputs ####")
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=36)

    decoded_outputs = tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
    
    print("#### Calculating accuracy and cer score ####")

    labels_filepath = os.path.join(data_f, "labels_test_" + model_name + ".txt")
    outputs_filepath = os.path.join(data_f, "outputs_test_" + model_name + ".txt")

    # saving for later analysis
    with open(outputs_filepath, "w") as output:
        output.write(str(decoded_outputs))

    with open(labels_filepath, "w") as output:
        output.write(str(labels))

    # my_file = open(labels_filepath, "r")
    # labels = my_file.read().split(',')
    # my_file.close()

    # my_file = open(outputs_filepath, "r")
    # decoded_outputs = my_file.read().split(',')
    # my_file.close()

    idx_end = 100

    while idx_end != 700:
        # first get accuracy
        direct_acc = 0
        for i in range(idx_end):
            if decoded_outputs[i] == labels[i]: # not lower cause german nouns are always capital
                direct_acc += 1
        print(f"Direct accuracy (%) on test set of size {idx_end}: {((direct_acc/idx_end) * 100)}")

        # then get CER
        metric = evaluate.load("cer")
        cer_score = metric.compute(predictions=decoded_outputs[0:idx_end], references=labels[0:idx_end])    
        print(f"CER on test set of size {idx_end}: {cer_score}") # lower is better
        print()
        idx_end += 100

def get_gender_accuracy(file_path_labels, file_path_outputs, file_path_testset):
    """ Get accuracy based on gender specifics (for whole test set)
    """
    my_file = open(file_path_labels, "r")
    labels = my_file.read().split(',')
    my_file.close()

    my_file = open(file_path_outputs, "r")
    outputs = my_file.read().split(',')
    my_file.close()

    tags = pd.read_csv(file_path_testset, usecols = ['tag'], low_memory = True)
    tags = tags['tag'].values.tolist()

    fem_labels = []
    masc_labels = []
    neut_labels = []

    fem_outputs = []
    masc_outputs = []
    neut_outputs = []

    for i in range(len(tags)):
        if "FEM" in tags[i]:
            fem_labels.append(labels[i])
            fem_outputs.append(outputs[i])
        elif "MASC" in tags[i]:
            masc_labels.append(labels[i])
            masc_outputs.append(outputs[i])
        else:
            neut_labels.append(labels[i])
            neut_outputs.append(outputs[i])

    fem_acc = 0
    masc_acc = 0
    neut_acc = 0

    for i in range(len(fem_outputs)):
        if fem_outputs[i] == fem_labels[i]:
            fem_acc += 1

    for i in range(len(masc_outputs)):
        if masc_outputs[i] == masc_labels[i]:
            masc_acc += 1

    for i in range(len(neut_outputs)):
        if neut_outputs[i] == neut_labels[i]:
            neut_acc += 1

    print(f"Accuracy on FEM nouns: {((fem_acc/len(fem_outputs)) * 100)}")
    print(f"Accuracy on MASC nouns: {((masc_acc/len(masc_outputs)) * 100)}")
    print(f"Accuracy on NEUT nouns: {((neut_acc/len(neut_outputs)) * 100)}")

def cer_per_length(file_path_labels, file_path_outputs, file_path_testset):
    my_file = open(file_path_labels, "r")
    labels = my_file.read().split(',')
    my_file.close()

    my_file = open(file_path_outputs, "r")
    outputs = my_file.read().split(',')
    my_file.close()

    df = pd.read_csv(file_path_testset, usecols = ['tag','lemma'], low_memory = True)
    tags = df['tag'].values.tolist()
    lemma = df['lemma'].values.tolist()

    metric = evaluate.load("cer")

    inputs = []
    unique_lens = []
    cer_scores = []
    acc_scores = []
    for i in range(len(tags)):
        input_ = tags[i] + ": " + lemma[i]
        inputs.append(input_)
        if len(input_) not in unique_lens:
            unique_lens.append(len(input_))

    unique_lens.sort()

    for length in unique_lens:
        indices = [i for i, x in enumerate(inputs) if len(x) == length]
        preds = [outputs[x] for x in indices]
        refs = [labels[x] for x in indices]
        acc = 0
        cer_score = metric.compute(predictions=preds, references=refs)
        for i in range(len(refs)):
            if preds[i] == refs[i]:
                acc += 1
        acc = ((acc/len(refs)) * 100)
        cer_scores.append(round(cer_score,2))
        acc_scores.append(round(acc, 2))
        print(f"CER score for input length {length}: {cer_score}")
        print(f"Direct accuracy for input length {length}: {acc}")
        print()

    X_axis = np.arange(len(unique_lens))
      
    plt.bar(X_axis, cer_scores)      
    plt.xticks(X_axis, unique_lens)
    plt.xlabel("Input lengths")
    plt.ylabel("CER")
    plt.title("CER scores for varying input lengths")
    plt.savefig("cer.png", format="png",
                dpi=300, bbox_inches="tight")

    plt.bar(X_axis, acc_scores)      
    plt.xticks(X_axis, unique_lens)
    plt.xlabel("Input lengths")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy scores for varying input lengths")
    plt.savefig("accuracy.png", format="png",
                dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    import pandas as pd
    # test purpose
    cwd = os.getcwd()
    file_path = os.path.dirname(os.path.dirname(cwd))
    labels = os.path.join(file_path, "german_data", "labels_test.txt")
    outputs = os.path.join(file_path, "german_data", "outputs_test.txt")
    testset = os.path.join(file_path, "german_data", "deu_gold.csv")
    
    get_gender_accuracy(labels, outputs, testset)
    cer_per_length(labels, outputs, testset)
