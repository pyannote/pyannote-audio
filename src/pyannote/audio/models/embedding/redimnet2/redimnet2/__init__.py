from .redimnet2 import ReDimNet2Wrap, ReDimNet2Custom

import os
import torch


URL_TEMPLATE = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{model_name}-vox2-{train_type}.pt"


def from_pretrained(model_name='b0', train_type='lm'):
    """Load a pretrained ReDimNet2 model.

    Args:
        model_name: One of 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6'.
        train_type: 'ptn' (pretraining) or 'lm' (large-margin fine-tuning).

    Returns:
        ReDimNet2Wrap model with loaded weights.
    """
    url = URL_TEMPLATE.format(model_name=model_name, train_type=train_type)
    full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']
    model = ReDimNet2Wrap(**model_config)
    load_res = model.load_state_dict(state_dict)
    assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0
    return model


class ReDimNet2:
    """Namespace for from_pretrained access."""
    from_pretrained = staticmethod(from_pretrained)
