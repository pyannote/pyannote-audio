import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from redimnet2.redimnet2 import ReDimNet2Wrap

dependencies = ['torch', 'torchaudio']

URL_TEMPLATE = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{model_name}"


def load_custom(model_name='b0', train_type='lm', dataset='vox2'):
    model_name = f'{model_name}-{dataset}-{train_type}.pt'
    url = URL_TEMPLATE.format(model_name=model_name)
    full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']
    model = ReDimNet2Wrap(**model_config)
    load_res = model.load_state_dict(state_dict)
    assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0
    return model


def redimnet2(model_name='b0', train_type='lm', dataset='vox2', pretrained=True):
    """Load a ReDimNet2 speaker embedding model.

    Args:
        model_name: One of 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6'.
        train_type: 'ptn' (pretraining) or 'lm' (large-margin fine-tuning).
        dataset: Training dataset, default 'vox2'.
        pretrained: If True, load pretrained weights.

    Returns:
        ReDimNet2Wrap model.
    """
    if pretrained:
        return load_custom(model_name, train_type=train_type, dataset=dataset)
    raise ValueError("Only pretrained=True is supported. Use ReDimNet2Wrap directly for custom models.")
