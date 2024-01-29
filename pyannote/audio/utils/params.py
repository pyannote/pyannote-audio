# TODO - make it depth-recursive
# TODO - switch to Omegaconf maybe?

from typing import Optional


def merge_dict(defaults: dict, custom: Optional[dict] = None):
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params
