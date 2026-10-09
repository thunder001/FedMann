import torch


def model_aggregate(models, device, weighting='sample_size'):
    """
    Aggregate model parameters from individual trained models using a weighted average.
    Args:
        models (list): List of trained parameters and sample sizes for models.
        device (str): Device to perform computations on (e.g., 'cpu' or 'cuda').
        weighting (str): 'sample_size' (default) weights each site's parameters by its
            share of the total training sample size (nk/N, i.e. FedAvg). 'equal' weights
            every site equally (1/K), useful for testing whether a smaller site's
            contribution is being dominated by a larger site under sample-size weighting.
    Returns:
        Aggregated model parameters.
    """
    if weighting not in ('sample_size', 'equal'):
        raise ValueError("weighting must be 'sample_size' or 'equal'.")

    # breakpoint()
    aggregated_params = None
    total_samples = sum(model['sample_size'] for model in models)
    if total_samples == 0:
        raise ValueError("Total sample size is zero, cannot aggregate models.")
    num_models = len(models)
    for i, model in enumerate(models):
        if weighting == 'sample_size':
            weight = model['sample_size'] / total_samples
        else:
            weight = 1 / num_models
        model_params = model['model_state_dict']
        model_params = {k: v.to(device) for k, v in model_params.items()}
        # Aggregate parameters using a weighted average of model parameters
        if aggregated_params is None:
            aggregated_params = {k: v * weight for k, v in model_params.items()}
        else:
            for k in aggregated_params.keys():
                aggregated_params[k] += model_params[k] * weight
        # Ensure parameters are on the correct device
        aggregated_params = {k: v.to(device) for k, v in aggregated_params.items()}

    return aggregated_params
