import time, sys
from dataclasses import dataclass, field
from config import ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION


@dataclass
class Alert:
    level: str; class_name: str; distance_m: float
    fog_score: float; timestamp: float = field(default_factory=time.time)


class AlertManager:
    def __init__(self, audio=False):
        self._cooldowns = {"CRITICAL": 1.0, "WARNING": 3.0, "CAUTION": 6.0}
        self._last: dict[str, float] = {}
        self.audio = audio

    def evaluate(self, detections, fog_score) -> list[Alert]:
        candidates = []
        for d in detections:
            dist, cls = d["distance_m"], d["class_name"]
            if   dist <= ZONE_CRITICAL: lvl = "CRITICAL"
            elif dist <= ZONE_WARNING:  lvl = "WARNING"
            elif dist <= ZONE_CAUTION:  lvl = "CAUTION"
            else: continue
            candidates.append(Alert(lvl, cls, dist, fog_score))
        candidates.sort(key=lambda a: a.distance_m)
        fired = []
        for a in candidates:
            key = f"{a.level}:{a.class_name}"
            if time.time() - self._last.get(key, 0) >= self._cooldowns[a.level]:
                self._last[key] = time.time()
                fired.append(a)
                pfx = {"CRITICAL":"[!!!]","WARNING":"[WRN]","CAUTION":"[CAU]"}[a.level]
                print(f"{pfx} {a.class_name} @ {a.distance_m:.1f}m "
                      f"fog={a.fog_score:.2f}", file=sys.stderr)
        return fired
