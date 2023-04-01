import inseq
import pandas as pd
import tensorflow as tf
import torch 
import numpy as np
from inseq.data.aggregator import SequenceAttributionAggregator
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
    print(f"#### Doing analysis at model in filepath {model_filepath} ####")
    model = inseq.load_model(model_filepath, "input_x_gradient")
    out = model.attribute(
        input_text,
        attribute_target=True,
        step_scores=["probability"]
    )

    out.show()

def get_average_score(model_f, test_f, bCorrect = True):
    """ Challenge part, get the average attribution scores given a prediction list of
    plural nouns 
    :param bCorrect: if True get scores for correct predictions else get scores for incorrect predictions 
    """  
    pattern1 = ";FEM;|;MASC;|;NEUT;"    # gender              

    # get all data needed
    print("#### Getting data ####")
    df = pd.read_csv(test_f, low_memory = True)

    tags = df['tag'].values.tolist()
    lemma = df['lemma'].values.tolist()

    labels = df["target_form"].values.tolist()

    inputs = []
    for i in range(len(tags)):
        inputs.append(tags[i] + ": " + lemma[i])

    # Challange part
    model = inseq.load_model(model_f, "input_x_gradient")

    print("#### Attributing ####")
    extract_atts_fem = []
    extract_atts_masc = []
    extract_atts_neut = []

    print(len(inputs))

    for i in range(len(inputs)):

        # get attribution scores
        out = model.attribute(
            inputs[i],
            attribute_target=True,
            step_scores=["probability"]
        )   
        print(i)
        attribution = out.sequence_attributions[0]
        target = ''.join([t.token for t in attribution.target][:-1])    # ignore eos token

        if (target == labels[i])  == bCorrect:
            start_idx, end_idx = re.search(pattern1, inputs[i]).span()
            # Aggregate scores to get the values from the heatmaps
            agg = out.sequence_attributions[0].aggregate(aggregator=SequenceAttributionAggregator)
            source_att = agg.source_attributions.numpy()
            # extract statistics around ;GEND;
            extract_att = source_att[np.arange(start_idx, end_idx), :]
            extract_att = extract_att[:,[-4,-3,-2]]     # ignore eos token (no -1)
            # save for analysis
            if "FEM" in inputs[i]:
                extract_atts_fem.append(extract_att)
            if "MASC" in inputs[i]:
                extract_atts_masc.append(extract_att)
            if "NEUT" in inputs[i]:
                extract_atts_neut.append(extract_att)

    # averaging 
    # all genders have both correct and inccorect predictions in test set
    # so not checking for nan values
    fem_matrix = np.average(extract_atts_fem, axis=0)
    masc_matrix = np.average(extract_atts_masc, axis=0)
    neut_matrix = np.average(extract_atts_neut, axis=0)
    print(f"average attribution score for fem for last 3 tokens: \n{np.around(fem_matrix,2)}\n")
    print(f"average attribution score for masc for last 3 tokens: \n{np.around(masc_matrix, 2)}\n")
    print(f"average attribution score for neut for last 3 tokens: \n{np.around(neut_matrix, 2)}\n")

    # average of averages
    print(f"overall average for all fem suffix: {round(np.mean(fem_matrix), 2)}")
    print(f"overall average for all masc suffix: {round(np.mean(masc_matrix), 2)}")
    print(f"overall average for all neut suffix: {round(np.mean(neut_matrix), 2)}")
