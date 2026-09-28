"""
Alert Manager
=============
Handles:
  - Per-object cooldown logic (each object gets its own alert, no spam)
  - Priority-based ordering (closest object first)
  - Multi-channel dispatch: console, voice (pyttsx3), optional beep

Alert levels  ->  Zones:
    CRITICAL  : distance <= ZONE_CRITICAL   -> immediate danger
    WARNING   : distance <= ZONE_WARNING    -> slow down
    CAUTION   : distance <= ZONE_CAUTION    -> be aware

Why the old version only spoke about the first object
-----------------------------------------------------
1. The cooldown key was "LEVEL:class_name", so two people (or the same
   person seen again) shared ONE cooldown and only one alert fired.
2. Speech engines like pyttsx3 commonly stop responding after the first
   runAndWait() when the same engine object is reused. Here speech runs in
   its own worker thread and the engine is re-created for every message.
"""

import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import config as _cfg
from config import ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION

# Import the speech library once, on the main thread (importing it inside the
# worker thread can fail during start-up), and remember the real error if any.
try:
    import pyttsx3
    _PYTTSX3_ERR = None
except Exception as _e:
    pyttsx3 = None
    _PYTTSX3_ERR = _e

# Optional settings - use config.py values if present, otherwise defaults
VOICE_RATE   = getattr(_cfg, "VOICE_RATE", 165)
VOICE_VOLUME = getattr(_cfg, "VOICE_VOLUME", 1.0)

# Horizontal bucket width (pixels) used to tell apart objects of the same class.
# Two people more than ~this far apart horizontally count as separate objects.
OBJECT_BUCKET_PX = 120


@dataclass
class Alert:
    level: str           # "CRITICAL" | "WARNING" | "CAUTION"
    class_name: str
    distance_m: float
    fog_score: float
    key: str = ""
    timestamp: float = field(default_factory=time.time)


class VoiceSpeaker:
    """
    Non-blocking text-to-speech. say() returns immediately; a background
    thread speaks the message. If messages pile up, the OLDEST is dropped
    so speech never lags far behind the video.
    """

    def __init__(self, rate: int = VOICE_RATE, volume: float = VOICE_VOLUME):
        self._rate = rate
        self._volume = volume
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text: str):
        while True:
            try:
                self._q.put_nowait(text)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()      # drop stale message
                except queue.Empty:
                    pass

    def close(self):
        while True:
            try:
                self._q.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
        self._thread.join(timeout=2)

    def _worker(self):
        # COM must be initialised on this thread for Windows SAPI voices
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        if pyttsx3 is None:
            print(f"[Voice] pyttsx3 unavailable: {_PYTTSX3_ERR!r}", file=sys.stderr)
            return

        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                # Fresh engine per message: avoids the "speaks once, then
                # silent" problem of reusing one engine across runAndWait().
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[Voice] speech failed: {e}", file=sys.stderr)


class AlertManager:
    def __init__(self,
                 critical_cooldown: float = 1.0,
                 warning_cooldown:  float = 3.0,
                 caution_cooldown:  float = 6.0,
                 audio: bool = False,
                 voice: bool = True,
                 voice_levels: tuple = ("CRITICAL", "WARNING"),
                 max_spoken_per_frame: int = 3):
        """
        Args:
            *_cooldown           : seconds before the SAME object can alert again
            audio                : play a beep (winsound, Windows only)
            voice                : speak alerts aloud
            voice_levels         : which levels are spoken (CAUTION is quiet by default)
            max_spoken_per_frame : max objects announced in one spoken message
        """
        self._cooldowns = {
            "CRITICAL": critical_cooldown,
            "WARNING":  warning_cooldown,
            "CAUTION":  caution_cooldown,
        }
        self._last_fired: dict[str, float] = {}
        self._audio = audio
        self._history: list[Alert] = []

        self._voice_levels = set(voice_levels)
        self._max_spoken = max_spoken_per_frame
        self._speaker = VoiceSpeaker() if voice else None

    # -- Public API ---------------------------------------------------------

    def evaluate(self, detections: list[dict], fog_score: float) -> list[Alert]:
        """
        Evaluate detections and return the alerts that passed their cooldown.

        Args:
            detections : dicts with class_name, distance_m and (optionally)
                         box_xyxy, used to tell same-class objects apart
            fog_score  : current frame fog score (0-1)
        """
        candidates: list[Alert] = []
        for det in detections:
            d = det["distance_m"]
            c = det["class_name"]
            if d <= ZONE_CRITICAL:
                level = "CRITICAL"
            elif d <= ZONE_WARNING:
                level = "WARNING"
            elif d <= ZONE_CAUTION:
                level = "CAUTION"
            else:
                continue
            candidates.append(
                Alert(level, c, d, fog_score, key=self._object_key(level, det)))

        # closest first
        candidates.sort(key=lambda a: a.distance_m)

        now = time.time()
        fired: list[Alert] = []
        for alert in candidates:
            last = self._last_fired.get(alert.key, 0.0)
            if now - last >= self._cooldowns[alert.level]:
                self._last_fired[alert.key] = now
                self._history.append(alert)
                fired.append(alert)
                self._print(alert)
                if self._audio:
                    self._beep(alert.level)

        self._speak(fired)
        return fired

    def get_history(self) -> list[Alert]:
        return list(self._history)

    def close(self):
        if self._speaker is not None:
            self._speaker.close()

    # -- Private ------------------------------------------------------------

    @staticmethod
    def _object_key(level: str, det: dict) -> str:
        """One cooldown slot per object: class + rough horizontal position."""
        box = det.get("box_xyxy")
        if box is not None:
            cx = (float(box[0]) + float(box[2])) / 2.0
            return f"{level}:{det['class_name']}:{int(cx // OBJECT_BUCKET_PX)}"
        return f"{level}:{det['class_name']}"

    def _speak(self, fired: list[Alert]):
        """One spoken message per frame covering the closest fired objects."""
        if not self._speaker:
            return
        spoken = [a for a in fired if a.level in self._voice_levels]
        if not spoken:
            return

        prefix = {"CRITICAL": "Danger", "WARNING": "Warning", "CAUTION": "Caution"}
        parts = [
            f"{prefix[a.level]}, {a.class_name}, {a.distance_m:.1f} metres"
            for a in spoken[: self._max_spoken]
        ]
        self._speaker.say(". ".join(parts))

    @staticmethod
    def _print(alert: Alert):
        prefix = {
            "CRITICAL": "[!!!CRITICAL!!!]",
            "WARNING":  "[ WARNING ]",
            "CAUTION":  "[ caution ]",
        }[alert.level]
        fog_note = f" (fog: {alert.fog_score:.2f})" if alert.fog_score > 0.3 else ""
        print(f"{prefix}  {alert.class_name} detected at "
              f"{alert.distance_m:.1f} m{fog_note}", file=sys.stderr)

    @staticmethod
    def _beep(level: str):
        try:
            import winsound
            freq = {"CRITICAL": 1200, "WARNING": 800, "CAUTION": 500}[level]
            duration = {"CRITICAL": 400, "WARNING": 250, "CAUTION": 150}[level]
            winsound.Beep(freq, duration)
        except Exception:
            pass   # non-Windows -> silent fallback
