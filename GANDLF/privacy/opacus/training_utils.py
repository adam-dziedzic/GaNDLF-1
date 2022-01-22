import torch
import torchio

from opacus import PrivacyEngine


def handle_nonempty_batch(subject, params):
    """
    Function to detect batch size from the subject an opacus loader 
    provides in the case of a non-empty batch, and make any changes to 
    the subject dictionary that are needed for GANDLF to use it.

    Parameters
    ----------
    subject : dict
        Training data subject dictionary.
    params : dict
        Training parameters.

    Returns
    -------
    batch_size : int
        Number of samples in the subject batch.
    

    """
    batch_size = len(subject[params["channel_keys"][0]][torchio.DATA])
    return subject, batch_size


def handle_empty_batch(subject, params, feature_shape, label_shape):
    """
    Function to replace the list of empty arrays an opacus loader 
    provides in the case of an empty batch with a subject dictionary GANDLF can consume.

    Parameters
    ----------
    subject : dict
        Training data subject dictionary.
    params : dict
        Training parameters.
    feature_shape : list
        Shape of a single feature in a batch.
    label_shape: list
        Shape of single label in a batch.


    Returns
    -------
    subject : dict
        Modified subject dictionary Train loss for the current epoch
    

    """

    print("\nConstructing empty batch dictionary.\n")

    subject = {'subject_id': 'empty_batch',
               'spacing': None, 
               'path_to_metadata': None, 
               'value_0': None, 
               'location': None}
    subject.update({key: {torchio.DATA: torch.zeros(tuple([0] + feature_shape ))} for key in params["channel_keys"]})
    if "value_keys" in params:

        subject.update({key: torch.zeros((0, 3)).to(torch.int64) for key in params["value_keys"]})
    else:
        subject.update({"label": {torchio.DATA: torch.zeros(tuple([0] + label_shape)).to(torch.int64) }})
    
    return subject

def handle_dynamic_batch_size(subject, params, feature_shape, label_shape):
    # TODO: Replace hard-coded feature an label shapes above with info from config or other
    """
    Function to process the subject opacus loaders 
    provide and prepare to handle their dynamic batch size 
    (including possible empty batches).

    Parameters
    ----------
    subject : dict
        Training data subject dictionary.
    params : dict
        Training parameters.
    feature_shape : list
        Shape of a single feature in a batch.
    label_shape: list
        Shape of single label in a batch.


    Returns
    -------
    subject : dict
        Modified subject dictionary Train loss for the current epoch
    
    """

    # The handling performed here is currently to be able to comprehend what 
    # batch size we are currently working with (which we may later see as not needed)
    # and also to handle the previously observed case where opacus produces
    # a subject that is not a dictionary but rather a list of empty arrays
    # (due to the empty batch result). The latter case is detected as a subject that
    # is a list object.
    if isinstance(subject, list):
        are_empty = torch.Tensor([torch.equal(tensor, torch.Tensor([])) for tensor in subject])
        if  not torch.all(are_empty):
            raise RuntimeError("Detected a list subject that is not an empty batch. This is not expected behavior.")
        else:
            subject = handle_empty_batch(subject=subject, 
                                        params=params, 
                                        feature_shape=feature_shape, 
                                        label_shape=label_shape)
            batch_size = 0
    else:
        subject, batch_size = handle_nonempty_batch(subject=subject, 
                                                    params=params)
            
    return subject, batch_size


