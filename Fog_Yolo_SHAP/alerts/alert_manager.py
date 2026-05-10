"""
Alert Manager
=============
Handles:
  - Cooldown logic (avoid alert spam every frame)
  - Priority-based alert selection (closest / most dangerous first)
  - Multi-channel dispatch: console, on-screen banner, optional audio beep

Alert levels  →  Zones:
    CRITICAL  : distance ≤ ZONE_CRITICAL   → immediate danger
    WARNING   : distance ≤ ZONE_WARNING    → slow down
    CAUTION   : distance ≤ ZONE_CAUTION    → be aware
"""

import time
import sys
from dataclasses import dataclass, field
from config import ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION


@dataclass
class Alert:
    level: str           # "CRITICAL" | "WARNING" | "CAUTION"
    class_name: str
    distance_m: float
    fog_score: float
    timestamp: float = field(default_factory=time.time)


class AlertManager:
    def __init__(self,
                 critical_cooldown: float = 1.0,
                 warning_cooldown:  float = 3.0,
                 caution_cooldown:  float = 6.0,
                 audio: bool = False):
        """
        Args:
            critical_cooldown : seconds between repeated CRITICAL alerts
            warning_cooldown  : seconds between repeated WARNING alerts
            caution_cooldown  : seconds between repeated CAUTION alerts
            audio             : attempt to play a beep via playsound (optional)
        """
        self._cooldowns  = {
            "CRITICAL": critical_cooldown,
            "WARNING":  warning_cooldown,
            "CAUTION":  caution_cooldown,
        }
        self._last_fired: dict[str, float] = {}
        self._audio = audio
        self._history: list[Alert] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, detections: list[dict], fog_score: float) -> list[Alert]:
        """
        Evaluate a list of detections and return any alerts that should fire.

        Args:
            detections : list of dicts with keys: class_name, distance_m
            fog_score  : current frame fog score (0–1)

        Returns:
            list of Alert objects that passed the cooldown gate
        """
        candidates: list[Alert] = []
        for det in detections:
            d = det["distance_m"]
            c = det["class_name"]
            if d <= ZONE_CRITICAL:
                candidates.append(Alert("CRITICAL", c, d, fog_score))
            elif d <= ZONE_WARNING:
                candidates.append(Alert("WARNING", c, d, fog_score))
            elif d <= ZONE_CAUTION:
                candidates.append(Alert("CAUTION", c, d, fog_score))

        # Sort by distance (closest first), then severity
        candidates.sort(key=lambda a: a.distance_m)

        fired: list[Alert] = []
        for alert in candidates:
            key = f"{alert.level}:{alert.class_name}"
            last = self._last_fired.get(key, 0.0)
            cooldown = self._cooldowns[alert.level]
            if time.time() - last >= cooldown:
                self._last_fired[key] = time.time()
                self._history.append(alert)
                fired.append(alert)
                self._dispatch(alert)

        return fired

    def get_history(self) -> list[Alert]:
        return list(self._history)

    # ── Private ─────────────────────────────────────────────────────────────

    def _dispatch(self, alert: Alert):
        """Print to console and optionally beep."""
        prefix = {
            "CRITICAL": "[!!!CRITICAL!!!]",
            "WARNING":  "[ WARNING ]",
            "CAUTION":  "[ caution ]",
        }[alert.level]

        fog_note = f" (fog: {alert.fog_score:.2f})" if alert.fog_score > 0.3 else ""
        msg = (f"{prefix}  {alert.class_name} detected at "
               f"{alert.distance_m:.1f} m{fog_note}")
        print(msg, file=sys.stderr)

        if self._audio:
            self._beep(alert.level)

    @staticmethod
    def _beep(level: str):
        try:
            import winsound
            freq = {"CRITICAL": 1200, "WARNING": 800, "CAUTION": 500}[level]
            duration = {"CRITICAL": 400, "WARNING": 250, "CAUTION": 150}[level]
            winsound.Beep(freq, duration)
        except Exception:
            pass   # non-Windows or playsound not installed → silent fallback
