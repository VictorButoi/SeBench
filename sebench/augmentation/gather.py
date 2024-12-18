import albumentations
from ionpy.experiment.util import absolute_import


def augmentations_from_config(aug_config_dict_list):
    if aug_config_dict_list is None:
        return None
    else:
        return albumentations.Compose([
            build_aug(aug_cfg) for aug_cfg in aug_config_dict_list
        ], is_check_shapes=False)

def build_aug(aug_obj):
    if isinstance(aug_obj, dict):
        # Go through the keys of the aug_name_key of the object and convert from list to tuple if necessary
        aug_name_key = list(aug_obj.keys())[0]
        a_dict = aug_obj[aug_name_key]
        for key in a_dict:
            if isinstance(a_dict[key], list):
                a_dict[key] = tuple(a_dict[key])

        # TODO: Remove, this is a stop-gap for backwards compatibility.
        aug_name_key = aug_name_key.replace("ese.experiment", "ese")

        # Get the key.
        return absolute_import(aug_name_key)(**a_dict)
    else:
        return absolute_import(aug_obj)()
        


