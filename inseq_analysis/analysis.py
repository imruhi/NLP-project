import inseq
import pandas as pd
import tensorflow as tf
import torch 
import numpy as np
from inseq.data.aggregator import AggregatorPipeline, ContiguousSpanAggregator, SequenceAttributionAggregator, PairAggregator

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

    # TODO challenge part

def do_analysis(model_filepath, input_text):
    """ Do analysis of the prediction (unconstrained) as in W5E notebook
    """
    model = inseq.load_model(model_filepath, "input_x_gradient")
    out1 = model.attribute(
        input_text,
        attribute_target=True,
        step_scores=["probability"]
    )

    # TODO: challange part
    source = out1.sequence_attributions[0].source
    target = out1.sequence_attributions[0].target

    squeezesum = AggregatorPipeline([ContiguousSpanAggregator, SequenceAttributionAggregator])

    # Aggregate outputs to get the tables
    agg = out1.sequence_attributions[0].aggregate(aggregator=SequenceAttributionAggregator)
    source_att = agg.source_attributions
    target_att = target_attributions
    out1.show()

    print(source)
    print(source_att)
    print()
    print(target)
    print(target_att)
    # N;NOM;FEM;PL: Orgie