# TODO - make it depth-recursive
# TODO - switch to Omegaconf maybe?


def merge_dict(defaults: dict, custom: dict = None):
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params
