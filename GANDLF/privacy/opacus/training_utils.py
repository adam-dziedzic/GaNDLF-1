import torch
import torchio


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
    # TODO: measure batch size and modify subject dictionary if needed
    batch_size = len(subject[params["channel_keys"][0]][torchio.DATA])
    return (subject, batch_size)


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


    # TODO: Pass info below via params (or some other method?)
    # replace subject with empty subject. Also using torchio.DATA when it is not
    # a torch image :(

    # TODO: remove test below
    print("\nConstructing empty batch dictionary.\n")

    batch_size = 0
    subject = {'subject_id': 'empty_batch',
               'spacing': None, 
               'path_to_metadata': None, 
               'value_0': None, 
               'location': None}
    subject.update({key: {torchio.DATA: torch.zeros(tuple([0] + feature_shape ))} for key in params["channel_keys"]})
    if "value_keys" in params:

        # TODO: remove test below
        value_keys = params["value_keys"]
        print(f"Here are value_keys: {value_keys}")
        
        subject.update({key: torch.zeros((0, 3)).to(torch.float32) for key in params["value_keys"]})
    else:
        subject.update({"label": {torchio.DATA: torch.zeros(tuple([0] + label_shape)).to(torch.int64) }})
        
        # TODO: remove test below
        label_obj = subject["label"]
        print(f"DEBUG, subjectlabel(shape) is: {label_obj[torchio.DATA]}, {label_obj[torchio.DATA].shape}")

    return (subject, batch_size)

def handle_dynamic_batch_size(subject, params, feature_shape=[3, 128, 128], label_shape=[3]):
    # TODO: Replace hard-coded feature an label shapes above with info from config or other
    """
    Function to replace the subject opacus loaders 
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

    # TODO: remove test below
    print(f"List subject is: {subject} and subject[0] is: {subject[0]}")
    
    # here looking for the case of an empty bach returned from opacus
    are_empty = torch.Tensor([torch.equal(tensor, torch.Tensor([])) for tensor in subject])
    if  not torch.all(are_empty):
        (subject, batch_size) =  handle_nonempty_batch(subject=subject, params=params)
    else:
        subject = handle_empty_batch(subject=subject, 
                                     params=params, 
                                     feature_shape=feature_shape, 
                                     label_shape=label_shape)
        batch_size = 0
        
    return (subject, batch_size)
        
