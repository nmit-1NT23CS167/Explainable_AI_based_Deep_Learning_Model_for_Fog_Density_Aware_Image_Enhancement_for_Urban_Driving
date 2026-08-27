import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from config import (ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION,
                    VOICE_ALERTS, VOICE_RATE, VOICE_VOLUME)


@dataclass
class Alert:
    level: str; class_name: str; distance_m: float
    fog_score: float; timestamp: float = field(default_factory=time.time)


class AlertManager:
    def __init__(self, audio=VOICE_ALERTS):
        self._cooldowns = {"CRITICAL": 1.0, "WARNING": 3.0, "CAUTION": 6.0}
        self._last: dict[str, float] = {}
        self.audio = audio
        self._speech_queue = queue.Queue()
        self._speech_thread = None
        if self.audio:
            self._speech_thread = threading.Thread(
                target=self._speech_worker, name="voice-alerts", daemon=True)
            self._speech_thread.start()

    def _speech_worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", VOICE_RATE)
            engine.setProperty("volume", VOICE_VOLUME)
        except Exception as exc:
            print(f"[Voice alerts] Disabled: {exc}", file=sys.stderr)
            return

        while True:
            message = self._speech_queue.get()
            try:
                if message is None:
                    return
                engine.say(message)
                engine.runAndWait()
            except Exception as exc:
                print(f"[Voice alerts] {exc}", file=sys.stderr)
            finally:
                self._speech_queue.task_done()

    def close(self):
        """Stop the speech worker after already queued alerts are spoken."""
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_queue.put(None)
            self._speech_thread.join(timeout=3)

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
                if self.audio and self._speech_thread and self._speech_thread.is_alive():
                    self._speech_queue.put(
                        f"{a.class_name} at {a.distance_m:.1f} meters")
        return fired
