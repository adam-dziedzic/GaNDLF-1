from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def opacus_model_fix(model, params):
    """
    Function to detect components of the model that are not compatible with Opacus
    differentially private training, and replacing with compatible components
    or raising an exception when a fix cannot be handled by Opacus.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    params : dict
        Training parameters.


    Returns
    -------
    model : torch.nn.Module
        Model, with potentially some components replaced with ones comptible with Opacus
        to allow for Opacus to make the training differentially private


    """
    # use opacus to detect issues with model
    opacus_errors_detected = ModuleValidator.validate(model, strict=False)

    if not params["differential_privacy"]["allow_opacus_model_fix"]:
        if opacus_errors_detected != []:
            raise RuntimeError(
                f"Training parameters are set to not allow Opacus to try to fix incompatible model components, and the following issues were detected: {opacus_errors_detected}"
            )
    elif opacus_errors_detected != []:
        print(
            f"Allowing Opacus to try and patch the model due to the following issues: ",
            opacus_errors_detected,
        )
        print()
        model = ModuleValidator.fix(model)
        # If the fix did not work, raise an exception
        ModuleValidator.validate(model, strict=True)
    return model


def prep_for_opacus_training(model, optimizer, train_dataloader, params):
    """
    Function to replace model, optimizer, and data loader to allow for
    differentially private training.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    optimizer : torch.optim.Optimizer
        Model optimizer.
    train_data_loader : torch.utils.data.DataLoader
        Dataloader for training data
    params : dict
        Training parameters


    Returns
    -------
    model : torch.nn.Module
        Model to be trained.
    optimizer : torch.optim.Optimizer
        Model optimizer.
    train_data_loader : torch.utils.data.DataLoader
        Dataloader for training data


    """

    privacy_engine = PrivacyEngine(
        accountant=params["differential_privacy"]["accountant"],
        secure_mode=params["differential_privacy"]["secure_mode"],
    )

    if not "epsilon" in params["differential_privacy"]:
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=params["differential_privacy"]["sigma"],
            max_grad_norm=params["differential_privacy"]["max_grad_norm"],
        )
    else:
        (
            model,
            optimizer,
            train_dataloader,
        ) = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            max_grad_norm=params["differential_privacy"]["max_grad_norm"],
            epochs=params["differential_privacy"]["epochs"],
            target_epsilon=params["differential_privacy"]["epsilon"],
            target_delta=params["differential_privacy"]["delta"],
        )
    return model, optimizer, train_dataloader
