

def parse_opacus_params(params, initialize_key):
    """
    Function to set defaults and augment the parameters related to making a trained model differentially
    private with respect to the training data.

    Parameters
    ----------
    params : dict
        Training parameters.
    initialize_key : function
        Function to fill in value for a missing key.

    Returns
    -------
    params : dict
        Training parameters
    
    """
    

    if not isinstance(params["differential_privacy"], dict):
        print("WARNING: Non dictionary value for the key: 'differential_privacy' was used, replacing with default valued dictionary.")
        params["differential_privacy"] = {}
        # these are some defaults
    if "noise_multiplier" in params["differential_privacy"]:
        params["differential_privacy"]["sigma"] = params["differential_privacy"][
            "noise_multiplier"
        ]
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "sigma", 1.0
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "max_grad_norm", 1.0
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "accountant", "rdp"
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "secure_mode", False
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "allow_opacus_model_fix", False
    )
    # this is required when epsilon is defined
    if "epsilon" in params["differential_privacy"]:
        params["differential_privacy"] = initialize_key(
            params["differential_privacy"], "delta", 1e-5
        )
        params["differential_privacy"] = initialize_key(
            params["differential_privacy"], "epochs", 20
        )
    return params