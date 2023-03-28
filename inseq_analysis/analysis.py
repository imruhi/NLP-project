import inseq
import pandas as pd
import tensorflow as tf
import torch 
import numpy as np
from inseq.data.aggregator import AggregatorPipeline, SequenceAttributionAggregator
import re

def do_contrast_analysis(model_filepath, input_text, contrast_text, truth_text):
    """ Do contrastive analysis as shown in W5E notebook  
    """
    model = inseq.load_model(model_filepath, "input_x_gradient")

    # incorrect one 
    contrast = model.encode(contrast_text, as_targets=True)

    out = model.attribute(
        input_text, # input to model
        truth_text, # Fix the original target
        attributed_fn="contrast_prob_diff",
        # Also show the probability delta between the two options
        step_scores=["contrast_prob_diff", "probability"],
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
    )

    # Normally attributions are normalized to sum up to 1
    # Here we want to see how they contribute to the probability difference instead
    out.weight_attributions("contrast_prob_diff")
    out.show()

def do_analysis(model_filepath, input_text):
    """ Do analysis of the prediction (unconstrained) as in W5E notebook
    """
    model = inseq.load_model(model_filepath, "input_x_gradient")
    out = model.attribute(
        input_text,
        attribute_target=True,
        step_scores=["probability"]
    )

    out.show()

def get_average_score(preds_f, labels_f, test_f, bCorrect = True):
    """ TODO: challenge part, get the average attribution scores given a prediction list of
    plural nouns 
    :param bCorrect: if True get scores for correct predictions else get scores for incorrect predictions 
    """  

    # get all data needed
    print("#### Getting data ####")
    my_file = open(file_path_labels, "r")
    labels = my_file.read().split(',')
    my_file.close()

    my_file = open(file_path_outputs, "r")
    outputs = my_file.read().split(',')
    my_file.close()

    df = pd.read_csv(file_path_testset, usecols = ['tag','lemma'], low_memory = True)
    tags = df['tag'].values.tolist()
    lemma = df['lemma'].values.tolist()

    inputs = []
    for i in range(len(tags)):
        inputs.append(tags[i] + ": " + lemma[i])

    # TODO: challange part
    print("#### Attributing ####")
    out = model.attribute(
        inputs,
        attribute_target=True,
        step_scores=["probability"]
    )

    # get first source and target from preds
    attribution = out.sequence_attributions[0]
    source = ''.join([t.token for t in attribution.source])
    target = ''.join([t.token for t in attribution.target])

    squeezesum = AggregatorPipeline([SequenceAttributionAggregator])

    # Aggregate outputs to get the values from the heatmaps (first pred)
    # round up to 2 decimal places
    agg = out.sequence_attributions[0].aggregate(aggregator=SequenceAttributionAggregator)
    source_att = np.around(agg.source_attributions.numpy(), 2)
    target_att = np.around(agg.target_attributions.numpy(), 2)
    out.show()

    pattern1 = ";FEM;|;MASC;|;NEUT;"    # gender
    pattern2 = ";N;"                    # pos
    pattern3 = ";NOM;"                  # case
    pattern4 = ";PL: "                  # number
    
    start_idx, end_idx = re.search(pattern1, source).span()
    extract_att = source_att[np.arange(start_idx, end_idx), :]
    
    # extract the section near the ;GENDER; area
    print(extract_att)  
    # N;NOM;FEM;PL: Orgie