"""Webcam index mapping. These tests do not open a camera."""

from __future__ import annotations

from smartvision.cameras import VideoDevice, device_for_user_index, unavailable_message

_LAPTOP = "AVCaptureDeviceTypeBuiltInWideAngleCamera"
_EXTERNAL = "AVCaptureDeviceTypeExternalUnknown"


def _devices() -> list[VideoDevice]:
    """Laptop first, which is the order that used to hide the iPhone."""
    return [
        VideoDevice("FaceTime HD Camera", "laptop", _LAPTOP, False),
        VideoDevice("iPhone Camera", "phone", _EXTERNAL, True),
        VideoDevice("iPhone Desk View Camera", "desk", "AVCaptureDeviceTypeDeskViewCamera", False),
    ]


def test_index_zero_is_the_iphone_even_when_the_laptop_is_listed_first() -> None:
    chosen = device_for_user_index(0, _devices())
    assert chosen is not None
    assert chosen.unique_id == "phone"


def test_index_one_is_the_laptop_even_when_it_is_not_device_one() -> None:
    chosen = device_for_user_index(1, _devices())
    assert chosen is not None
    assert chosen.unique_id == "laptop"


def test_missing_iphone_does_not_fall_back_to_the_laptop() -> None:
    laptop_only = [VideoDevice("FaceTime HD Camera", "laptop", _LAPTOP, False)]
    assert device_for_user_index(0, laptop_only) is None
    chosen = device_for_user_index(1, laptop_only)
    assert chosen is not None
    assert chosen.unique_id == "laptop"
    assert "iPhone" in unavailable_message(0)
    assert "Laptop" in unavailable_message(1)


def test_desk_view_is_not_used_as_the_iphone_camera() -> None:
    devices = [
        VideoDevice("iPhone Desk View Camera", "desk", "AVCaptureDeviceTypeDeskViewCamera", False),
        VideoDevice("FaceTime HD Camera", "laptop", _LAPTOP, False),
    ]
    assert device_for_user_index(0, devices) is None
