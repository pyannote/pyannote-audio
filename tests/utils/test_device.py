import torch

from pyannote.audio.utils.device import (
    get_accelerator_device,
    get_accelerator_device_count,
    get_accelerator_device_type,
)


def test_get_accelerator_device():
    device = get_accelerator_device()
    assert isinstance(device, torch.device)
    assert device.type == get_accelerator_device_type()


def test_get_accelerator_device_count():
    count = get_accelerator_device_count()
    assert isinstance(count, int)
    if get_accelerator_device_type() == "cpu":
        assert count == 0
    else:
        assert count >= 1


def test_get_devices():
    from pyannote.audio.pipelines.utils import get_devices

    device_type = get_accelerator_device_type()

    devices = get_devices()
    assert len(devices) >= 1
    assert all(device.type == device_type for device in devices)

    (device,) = get_devices(needs=1)
    assert device.type == device_type

    assert len(get_devices(needs=3)) == 3


def test_parse_device():
    from pyannote.audio.__main__ import Device, parse_device

    assert parse_device(Device.AUTO) == get_accelerator_device()
    assert parse_device(Device.CPU) == torch.device("cpu")
    assert parse_device(Device.CUDA) == torch.device("cuda")
