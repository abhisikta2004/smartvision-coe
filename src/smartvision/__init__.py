"""SmartVision: context-aware PPE compliance and danger-zone monitoring.

Pipeline:

1. Input frame (numpy array; a camera is optional).
2. COCO person detector plus the fine-tuned PPE model.
3. Person tracking at the bottom-center of the person box.
4. Danger-zone check: any overlap between the person box and a polygon counts. The highest risk wins.
5. PPE assignment by body region, compared with the zone's required list
   or the default list outside a zone.
6. IoU worker tracking so identity stays stable.
7. Amber warning outside a zone. Inside a zone, a hold timer then one
   critical alert, an evidence frame, and a per-worker cooldown.
8. Overlay and KPIs.
"""

from smartvision.alerts import Alert, Observation, ViolationManager
from smartvision.compliance import ComplianceEngine, ComplianceResult
from smartvision.detector import Detection, PPEDetector
from smartvision.pipeline import FrameResult, Pipeline, SmartVisionPipeline, Worker, WorkerStatus
from smartvision.tracker import IoUTracker, Track
from smartvision.visualize import FrameVisualizer
from smartvision.zones import Point, Zone, ZoneChecker, bottom_center, find_zone, load_zones, person_anchor, save_zones

__version__ = "0.2.0"

__all__ = [
    "Alert",
    "ComplianceEngine",
    "ComplianceResult",
    "Detection",
    "FrameResult",
    "FrameVisualizer",
    "IoUTracker",
    "Observation",
    "PPEDetector",
    "Pipeline",
    "Point",
    "SmartVisionPipeline",
    "Track",
    "ViolationManager",
    "Worker",
    "WorkerStatus",
    "Zone",
    "ZoneChecker",
    "bottom_center",
    "find_zone",
    "load_zones",
    "person_anchor",
    "save_zones",
    "__version__",
]
