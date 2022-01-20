import torch
import torchio


def handle_dynamic_batch_size(subject, params, feature_shape=[3, 128, 128], label_shape=[3]):
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


    if not isinstance(subject, list) or params["differential_privacy"] is None:
        return subject
    else:
        print(f"List subject is: {subject} and subject[0] is: {subject[0]}")
        # here capturing the case of an empty bach returned from opacus
    
        are_empty = torch.Tensor([torch.equal(tensor, torch.Tensor([])) for tensor in subject])
        if  not torch.all(are_empty):
            return subject
        else:
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
                thingy = params["value_keys"]
                print(f"Here are value_keys: {thingy}")
                
                subject.update({key: torch.zeros((0, 3)).to(torch.float32) for key in params["value_keys"]})
            else:
                subject.update({"label": {torchio.DATA: torch.zeros(tuple([0] + label_shape)).to(torch.int64) }})
                # TODO: remove test below
                thingout = subject["label"]
                print(f"DEBUG, subjectlabel(shape) is: {thingout[torchio.DATA]}, {thingout[torchio.DATA].shape}")
            return subject
