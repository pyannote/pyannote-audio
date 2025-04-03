import pytest
import torch
from unittest.mock import patch, MagicMock

from pyannote.audio.core.model import Model
from pyannote.audio.pipelines.utils.getter import get_model


class BrokenModelWithoutEval(Model):
    def __init__(self):
        super().__init__()
        self.eval = None
    
    def forward(self, waveforms):
        return torch.rand(1)


class BrokenModelWithNonCallableEval(Model):
    def __init__(self):
        super().__init__()
        self.eval = "not_callable"
    
    def forward(self, waveforms):
        return torch.rand(1)


def test_model_without_eval_attribute():
    model = BrokenModelWithoutEval()
    
    with patch('pyannote.audio.pipelines.utils.getter.hasattr', return_value=False):
        with pytest.raises(ValueError) as excinfo:
            get_model(model)
        
        assert "The model could not be loaded" in str(excinfo.value)
        assert f"Recieved: {model}" in str(excinfo.value)


def test_model_with_non_callable_eval():
    model = BrokenModelWithNonCallableEval()
    
    with pytest.raises(ValueError) as excinfo:
        get_model(model)
    
    assert "The model could not be loaded" in str(excinfo.value)
    assert f"Recieved: {model}" in str(excinfo.value)


@patch('pyannote.audio.core.model.Model.from_pretrained')
def test_get_model_with_auth_token(mock_from_pretrained):
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_from_pretrained.return_value = mock_model
    
    model_path = "dummy/model/path"
    auth_token = "test_token"
    result = get_model(model_path, use_auth_token=auth_token)
    
    mock_from_pretrained.assert_called_once_with(
        model_path, use_auth_token=auth_token, strict=False
    )
    
    mock_model.eval.assert_called_once()
    
    assert result == mock_model


@patch('pyannote.audio.core.model.Model.from_pretrained')
def test_get_model_with_dict_config(mock_from_pretrained):
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_from_pretrained.return_value = mock_model
    
    model_config = {
        "checkpoint": "dummy/model/path",
        "map_location": "cuda:0"
    }
    auth_token = "test_token"
    result = get_model(model_config, use_auth_token=auth_token)
    
    expected_config = model_config.copy()
    expected_config["use_auth_token"] = auth_token
    
    mock_from_pretrained.assert_called_once_with(**expected_config)
    
    mock_model.eval.assert_called_once()
    
    assert result == mock_model


def test_get_model_with_invalid_type():
    with pytest.raises(TypeError) as excinfo:
        get_model(42)
    
    assert "Unsupported type" in str(excinfo.value)
    assert "expected `str` or `dict`" in str(excinfo.value)
