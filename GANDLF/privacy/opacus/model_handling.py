from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def opacus_model_fix(model, params):
    # use opacus to detect issues with model
    opacus_errors_detected = ModuleValidator.validate(model, strict=False)

    if not params["differential_privacy"]["allow_opacus_model_fix"]:
        if opacus_errors_detected != []:
            raise RuntimeError(f"Training parameters are set to not allow Opacus to try to fix incompatible model components, and the following issues were detected: {opacus_errors_detected}")
    elif opacus_errors_detected != []:
        print(f"Allowing Opacus to try and patch the model due to the following issues: ",opacus_errors_detected)
        print()
        model = ModuleValidator.fix(model)
        # If the fix did not work, raise an exception
        ModuleValidator.validate(model, strict=True)