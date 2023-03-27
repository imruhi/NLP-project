import inseq

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
    # not very good probably because of lack of training data?
    # possibly could be because we're not tokenizing the tags and lemma correctly but don't know yet
    model = inseq.load_model(model_filepath, "input_x_gradient")
    out = model.attribute(
        input_text,
        attribute_target=True,
        step_scores=["probability"]
    )
    out.show()