"""Mac webcam selection.

OpenCV asks AVFoundation for cameras with the deprecated device list, which
on this Mac contains only the built-in FaceTime camera. The iPhone
Continuity Camera is visible to the modern discovery session, so index 0
would open the laptop and index 1 would fail.

User index 0 is the iPhone camera. User index 1 is the built-in laptop
camera. This module opens those devices itself and returns frames as BGR
numpy arrays. No function opens a camera unless a capture is constructed.
"""

from __future__ import annotations

import ctypes
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

_BUILTIN = "AVCaptureDeviceTypeBuiltInWideAngleCamera"


@dataclass(frozen=True)
class VideoDevice:
    """One camera reported by AVFoundation.

    Attributes:
        name: System name, such as the FaceTime camera or an iPhone.
        unique_id: Stable device id.
        device_type: AVFoundation device-type constant.
        continuity: ``True`` when this is a Continuity Camera.
        av_device: Underlying device object used to open a capture.
    """

    name: str
    unique_id: str
    device_type: str
    continuity: bool
    av_device: Any = field(default=None, compare=False, repr=False)

    @property
    def iphone(self) -> bool:
        """The iPhone camera, not its Desk View companion."""
        return self.continuity and "desk view" not in self.name.lower()

    @property
    def laptop(self) -> bool:
        """The built-in laptop camera."""
        return not self.continuity and self.device_type == _BUILTIN


def device_for_user_index(user_index: int, devices: Sequence[VideoDevice]) -> VideoDevice | None:
    """Map the dashboard index onto a physical camera.

    Args:
        user_index: ``0`` is the iPhone, ``1`` is the laptop webcam.
            Higher indexes select that position in ``devices``.
        devices: Cameras from :func:`list_video_devices`.

    Returns:
        The matching device, or ``None`` when it is not connected.
    """
    if user_index < 0:
        return None
    if user_index == 0:
        return next((device for device in devices if device.iphone), None)
    if user_index == 1:
        return next((device for device in devices if device.laptop), None)
    if user_index < len(devices):
        return devices[user_index]
    return None


def unavailable_message(user_index: int) -> str:
    """Explain a missing webcam.

    Args:
        user_index: Index the caller asked to open.

    Returns:
        A sentence suitable for the dashboard.
    """
    if user_index == 0:
        return (
            "iPhone camera is not available. Unlock the iPhone, turn on "
            "Continuity Camera, and keep it near this Mac."
        )
    if user_index == 1:
        return "Laptop webcam is not available."
    return f"Could not open camera index {user_index}."


@dataclass
class _DeviceWatch:
    """Queue of camera-list requests served on the discovery thread."""

    requests: queue.Queue[queue.Queue[list[VideoDevice] | BaseException]]
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


_device_watch: _DeviceWatch | None = None
_device_watch_lock = threading.Lock()
_device_watch_kept: tuple[Any, Any] | None = None


def _ensure_device_watch() -> _DeviceWatch:
    """Start the discovery thread once. Importing this module does not start it."""
    global _device_watch
    with _device_watch_lock:
        if _device_watch is None:
            _device_watch = _DeviceWatch(requests=queue.Queue())
            threading.Thread(
                target=_device_watch_loop,
                args=(_device_watch,),
                name="smartvision-cameras",
                daemon=True,
            ).start()
        return _device_watch


def _device_watch_loop(watch: _DeviceWatch) -> None:
    """Own the discovery session and answer list requests from that thread."""
    global _device_watch_kept
    try:
        avfoundation = _avfoundation()
        from Foundation import NSDate, NSNotificationCenter, NSObject, NSRunLoop
        import objc

        class _DeviceNotifier(NSObject):
            """Keep Continuity Camera connect and disconnect events flowing."""

            def init(self) -> _DeviceNotifier:
                self = objc.super(_DeviceNotifier, self).init()
                return self

            def devicesChanged_(self, _notification: object) -> None:
                return None

        notifier = _DeviceNotifier.alloc().init()
        center = NSNotificationCenter.defaultCenter()
        for name in (
            avfoundation.AVCaptureDeviceWasConnectedNotification,
            avfoundation.AVCaptureDeviceWasDisconnectedNotification,
        ):
            center.addObserver_selector_name_object_(notifier, "devicesChanged:", name, None)
        session = _discovery_session(avfoundation)
        _device_watch_kept = (notifier, session)
    except Exception as exc:
        watch.error = exc
        watch.ready.set()
        return
    watch.ready.set()
    while True:
        _drain_device_requests(watch, session)
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))


def _discovery_session(avfoundation: Any) -> Any:
    """Create the session that includes the iPhone and its Desk View camera."""
    types = [
        avfoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera,
        avfoundation.AVCaptureDeviceTypeExternalUnknown,
        avfoundation.AVCaptureDeviceTypeDeskViewCamera,
    ]
    return avfoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types,
        avfoundation.AVMediaTypeVideo,
        avfoundation.AVCaptureDevicePositionUnspecified,
    )


def _drain_device_requests(watch: _DeviceWatch, session: Any) -> None:
    """Answer every waiting camera-list request from the discovery thread."""
    while True:
        try:
            reply = watch.requests.get_nowait()
        except queue.Empty:
            return
        try:
            reply.put(_devices_from_session(session))
        except Exception as exc:
            reply.put(exc)


def _devices_from_session(session: Any) -> list[VideoDevice]:
    """Copy the current discovery-session devices."""
    devices: list[VideoDevice] = []
    for device in session.devices():
        devices.append(
            VideoDevice(
                name=str(device.localizedName()),
                unique_id=str(device.uniqueID()),
                device_type=str(device.deviceType()),
                continuity=bool(device.isContinuityCamera()),
                av_device=device,
            )
        )
    return devices


def list_video_devices() -> list[VideoDevice]:
    """Return the Mac cameras, including a connected iPhone.

    The discovery session stays alive on a run-loop thread. A session created
    once and discarded misses an iPhone that connects after this process
    started, and later scans in that process keep returning the old list.

    Returns:
        Discovery order. The iPhone is included when Continuity Camera is on.
    """
    watch = _ensure_device_watch()
    if not watch.ready.wait(30):
        raise RuntimeError("The camera list did not respond.")
    if watch.error is not None:
        raise RuntimeError(str(watch.error))
    reply: queue.Queue[list[VideoDevice] | BaseException] = queue.Queue()
    watch.requests.put(reply)
    result = reply.get(timeout=30)
    if isinstance(result, BaseException):
        raise result
    return result


def open_webcam(user_index: int) -> CameraCapture:
    """Open user index 0 as the iPhone and user index 1 as the laptop.

    Args:
        user_index: Webcam index from the dashboard or ``run_video``.

    Returns:
        A capture with ``read``, ``get``, ``isOpened``, and ``release``.

    Raises:
        RuntimeError: That camera is not connected.
    """
    device = device_for_user_index(user_index, list_video_devices())
    if device is None or device.av_device is None:
        raise RuntimeError(unavailable_message(user_index))
    return CameraCapture(device)


class CameraCapture:
    """BGR frames from one AVFoundation camera."""

    def __init__(self, device: VideoDevice) -> None:
        """Start ``device``.

        Args:
            device: A camera from :func:`list_video_devices`.
        """
        self.open_error: str | None = None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._condition = threading.Condition()
        self._opened = False
        self._fps = 30.0
        self._session: Any = None
        self._delegate: _SampleBufferDelegate | None = None
        self._output: Any = None
        try:
            self._start(device)
        except Exception as exc:
            self.open_error = f"Could not open {device.name}: {exc}"
            self.release()

    def isOpened(self) -> bool:
        """Report whether the capture session is running."""
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next BGR frame.

        Returns:
            ``(True, frame)`` or ``(False, None)`` when no frame arrives.
        """
        with self._condition:
            seen = self._sequence
            timeout = 8.0 if seen == 0 else 2.0
            self._condition.wait_for(lambda: self._sequence != seen, timeout)
            if self._frame is None:
                self.open_error = self.open_error or "The camera did not deliver a frame."
                return False, None
            return True, self._frame.copy()

    def get(self, prop: int) -> float:
        """Return a capture property. Only the frame rate is known.

        Args:
            prop: An OpenCV ``CAP_PROP_*`` value.

        Returns:
            Frames per second, or ``0`` for anything else.
        """
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def release(self) -> None:
        """Stop the session and drop the latest frame."""
        self._opened = False
        output = self._output
        session = self._session
        delegate = self._delegate
        self._output = None
        self._session = None
        self._delegate = None
        if delegate is not None:
            delegate.capture = None
        if output is not None:
            output.setSampleBufferDelegate_queue_(None, None)
        if session is not None:
            session.stopRunning()
        with self._condition:
            self._frame = None
            self._condition.notify_all()

    def _publish(self, frame: np.ndarray) -> None:
        """Store one frame for the next :meth:`read`."""
        with self._condition:
            self._frame = frame
            self._sequence += 1
            self._condition.notify_all()

    def _start(self, device: VideoDevice) -> None:
        """Configure and start an AVFoundation session."""
        avfoundation = _avfoundation()
        session = avfoundation.AVCaptureSession.alloc().init()
        self._session = session
        session.beginConfiguration()
        for preset in (
            avfoundation.AVCaptureSessionPreset1280x720,
            avfoundation.AVCaptureSessionPresetHigh,
            avfoundation.AVCaptureSessionPreset640x480,
        ):
            if session.canSetSessionPreset_(preset):
                session.setSessionPreset_(preset)
                break
        device_input, error = avfoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(
            device.av_device,
            None,
        )
        if device_input is None or not session.canAddInput_(device_input):
            detail = "" if error is None else f" {error}"
            raise RuntimeError(f"could not add {device.name}.{detail}")
        session.addInput_(device_input)
        output = avfoundation.AVCaptureVideoDataOutput.alloc().init()
        self._output = output
        from Quartz.CoreVideo import kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA

        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_(
            {kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA}
        )
        delegate = _SampleBufferDelegate.alloc().initWithCapture_(self)
        self._delegate = delegate
        output.setSampleBufferDelegate_queue_(delegate, _dispatch_queue())
        if not session.canAddOutput_(output):
            raise RuntimeError(f"could not add a frame output for {device.name}")
        session.addOutput_(output)
        session.commitConfiguration()
        self._fps = _frame_rate(device.av_device)
        session.startRunning()
        self._opened = bool(session.isRunning())
        if not self._opened:
            self.open_error = f"Could not open {device.name}."


class _SampleBufferDelegate:
    """Placeholder replaced after AVFoundation imports.

    The real class is created lazily so importing this module does not load
    the camera frameworks during tests that only check index mapping.
    """

    capture: CameraCapture | None = None

    def initWithCapture_(self, capture: CameraCapture) -> _SampleBufferDelegate:
        self.capture = capture
        return self


def _install_delegate() -> None:
    """Define the sample-buffer delegate once AVFoundation is loaded."""
    global _SampleBufferDelegate
    if getattr(_SampleBufferDelegate, "_av_ready", False):
        return
    import objc
    from Foundation import NSObject

    class _AVDelegate(NSObject):
        """Receive BGRA frames and hand BGR copies to the capture."""

        def initWithCapture_(self, capture: CameraCapture) -> _AVDelegate:
            self = objc.super(_AVDelegate, self).init()
            if self is None:
                return self
            self.capture = capture
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self,
            _output: object,
            sample_buffer: object,
            _connection: object,
        ) -> None:
            capture = getattr(self, "capture", None)
            if capture is None or sample_buffer is None:
                return
            try:
                frame = _frame_from_sample(sample_buffer)
            except Exception as exc:
                capture.open_error = f"{type(exc).__name__}: {exc}"
                return
            if frame is not None:
                capture._publish(frame)

    _AVDelegate._av_ready = True
    _SampleBufferDelegate = _AVDelegate


def _copy_pixels(base: object, size: int) -> bytes:
    """Copy ``size`` bytes from a pixel-buffer address."""
    if isinstance(base, int):
        return ctypes.string_at(base, size)
    as_buffer = getattr(base, "as_buffer", None)
    if callable(as_buffer):
        return bytes(as_buffer(size))
    for name in ("__c_void_p__", "ptr", "value"):
        pointer = getattr(base, name, None)
        if isinstance(pointer, int) and pointer:
            return ctypes.string_at(pointer, size)
    names = [name for name in dir(base) if not name.startswith("_")]
    raise TypeError(f"unsupported pixel address {type(base).__name__}: {names[:40]}")


def _frame_from_sample(sample_buffer: object) -> np.ndarray | None:
    """Copy one sample buffer into a BGR image."""
    import CoreMedia
    from Quartz.CoreVideo import (
        CVPixelBufferGetBaseAddress,
        CVPixelBufferGetBytesPerRow,
        CVPixelBufferGetHeight,
        CVPixelBufferGetWidth,
        CVPixelBufferLockBaseAddress,
        CVPixelBufferUnlockBaseAddress,
    )

    pixel = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
    if pixel is None:
        return None
    CVPixelBufferLockBaseAddress(pixel, 1)
    try:
        width = int(CVPixelBufferGetWidth(pixel))
        height = int(CVPixelBufferGetHeight(pixel))
        row_bytes = int(CVPixelBufferGetBytesPerRow(pixel))
        base = CVPixelBufferGetBaseAddress(pixel)
        if width <= 0 or height <= 0 or row_bytes < width * 4 or not base:
            return None
        raw = _copy_pixels(base, row_bytes * height)
    finally:
        CVPixelBufferUnlockBaseAddress(pixel, 1)
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes)[:, : width * 4]
    return cv2.cvtColor(bgra.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)


def _frame_rate(av_device: Any) -> float:
    """Read the camera's fastest advertised frame rate."""
    try:
        ranges = av_device.activeFormat().videoSupportedFrameRateRanges()
        if ranges:
            return float(ranges[0].maxFrameRate())
    except Exception:
        return 30.0
    return 30.0


def _dispatch_queue() -> object:
    """Create a serial queue for camera callbacks."""
    import objc

    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.dispatch_queue_create.restype = ctypes.c_void_p
    library.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    pointer = library.dispatch_queue_create(b"smartvision.camera", None)
    return objc.objc_object(c_void_p=pointer)


def _avfoundation() -> Any:
    """Import AVFoundation and install the frame delegate."""
    import AVFoundation

    _install_delegate()
    return AVFoundation
