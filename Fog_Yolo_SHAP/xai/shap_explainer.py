"""
SHAP Explainer for Fog + YOLO Distance Alert System
====================================================

What does SHAP explain here?
    The "model" SHAP analyses is a deterministic scoring function that maps
    per-detection features → alert_risk_score (0–1). This risk score combines:
        • estimated distance (closer = higher risk)
        • fog density (foggier = higher risk, because visibility is reduced)
        • detection confidence (lower confidence in fog = amplified risk)
        • bounding-box position (objects near image centre are in-path)

    SHAP KernelExplainer is model-agnostic: it doesn't need to open YOLO.
    It treats the risk function as a black box and measures each feature's
    marginal contribution using Shapley values from cooperative game theory.

SHAP output:
    shap_values : shape (n_samples, n_features)
        Positive value → pushes risk HIGHER
        Negative value → pushes risk LOWER
    base_value  : expected risk across the background dataset

Interpretation examples:
    fog_score = +0.18  → fog is adding 0.18 to the risk score for this detection
    distance  = -0.30  → this object is far enough to reduce risk by 0.30
    confidence= +0.05  → high confidence detection slightly raises risk
"""

import os
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

from config import (SHAP_BACKGROUND_SAMPLES, SHAP_MAX_EVALS,
                    SHAP_TOP_K_FEATURES, FEATURE_NAMES, OUTPUT_DIR)
from utils.distance_estimator import ZONE_CAUTION


# ── Risk scoring function ───────────────────────────────────────────────────────

def compute_risk_score(features: np.ndarray) -> np.ndarray:
    """
    Deterministic risk scoring function: feature vector → risk score in [0,1].
    This is the "model" that SHAP explains.

    Feature layout (matches FEATURE_NAMES):
        0  bbox_width_px
        1  bbox_height_px
        2  bbox_area_px
        3  center_x_norm
        4  center_y_norm
        5  confidence
        6  fog_score
        7  aspect_ratio

    Risk formula (interpretable by design for PoC):
        risk = α * distance_risk + β * fog_risk + γ * position_risk + δ * conf_risk
    """
    features = np.atleast_2d(features).astype(np.float32)
    risks = np.zeros(len(features))

    for i, f in enumerate(features):
        bbox_w      = max(f[0], 1.0)
        bbox_h      = max(f[1], 1.0)
        bbox_area   = f[2]
        cx_norm     = f[3]
        cy_norm     = f[4]
        confidence  = np.clip(f[5], 0.0, 1.0)
        fog_score   = np.clip(f[6], 0.0, 1.0)

        # Approximate distance from bbox width (rough inverse — for SHAP only)
        approx_dist = max(1.0, 1440.0 / bbox_w)   # 1440 = focal * typical_width

        # Component 1: distance risk — inverse, capped
        dist_risk = np.clip(1.0 - approx_dist / ZONE_CAUTION, 0.0, 1.0)

        # Component 2: fog risk — fog amplifies danger
        fog_risk = fog_score * (1.0 + dist_risk)
        fog_risk = np.clip(fog_risk, 0.0, 1.0)

        # Component 3: position risk — central lane objects more dangerous
        pos_risk = 1.0 - 2.0 * abs(cx_norm - 0.5)   # 1.0 at centre, 0 at edges
        pos_risk = np.clip(pos_risk, 0.0, 1.0)

        # Component 4: confidence — low confidence in fog → uncertain danger
        conf_risk = confidence * 0.2    # small weight

        risk = 0.45 * dist_risk + 0.30 * fog_risk + 0.15 * pos_risk + 0.10 * conf_risk
        risks[i] = float(np.clip(risk, 0.0, 1.0))

    return risks


# ── SHAP Explainer class ────────────────────────────────────────────────────────

class FogAlertSHAPExplainer:
    """
    Wraps SHAP KernelExplainer for the fog alert risk function.

    Usage:
        explainer = FogAlertSHAPExplainer(background_data)
        result = explainer.explain(feature_vector)
        explainer.plot_waterfall(result, save_path="output/shap/frame_042.png")
    """

    def __init__(self, background_data: np.ndarray):
        """
        Args:
            background_data : np.ndarray shape (N, 8) — representative sample
                              of feature vectors used as SHAP baseline.
        """
        self.background = background_data
        self.explainer = shap.KernelExplainer(
            model=compute_risk_score,
            data=self.background,
            link="identity",
        )
        self.feature_names = FEATURE_NAMES

    def explain(self, feature_vector: np.ndarray) -> dict:
        """
        Compute SHAP values for a single detection.

        Args:
            feature_vector : shape (8,) — one detection's features

        Returns:
            dict with keys:
                shap_values   : np.ndarray (8,)
                base_value    : float
                risk_score    : float
                top_features  : list[(name, shap_val)] sorted by |shap_val|
        """
        x = np.atleast_2d(feature_vector.astype(np.float32))
        sv = self.explainer.shap_values(x, nsamples=SHAP_MAX_EVALS, silent=True)
        sv = np.array(sv).flatten()

        risk = float(compute_risk_score(x)[0])
        base = float(self.explainer.expected_value)

        ranked = sorted(
            zip(self.feature_names, sv),
            key=lambda pair: abs(pair[1]),
            reverse=True
        )

        return {
            "shap_values":  sv,
            "base_value":   base,
            "risk_score":   risk,
            "top_features": ranked[:SHAP_TOP_K_FEATURES],
        }

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_waterfall(self, result: dict, title: str = "",
                       save_path: str = None) -> None:
        """
        Horizontal waterfall bar chart: shows each feature's SHAP contribution
        stacked from the base value toward the final risk score.
        """
        names  = [f[0] for f in result["top_features"]]
        values = [f[1] for f in result["top_features"]]
        base   = result["base_value"]
        risk   = result["risk_score"]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

        bars = ax.barh(names, values, color=colors, height=0.55)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        for bar, val in zip(bars, values):
            sign = "+" if val >= 0 else ""
            ax.text(val + (0.003 if val >= 0 else -0.003),
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=9)

        ax.set_xlabel("SHAP value  (contribution to risk score)")
        ax.set_title(
            f"{title}\nBase risk: {base:.3f}  →  Predicted risk: {risk:.3f}",
            fontsize=10
        )
        ax.set_xlim(-0.6, 0.6)

        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(color="#d73027", label="Increases risk"),
                  Patch(color="#4575b4", label="Decreases risk")]
        ax.legend(handles=legend, loc="lower right", fontsize=8)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")

        plt.close(fig)

    def plot_summary(self, feature_matrix: np.ndarray,
                     save_path: str = None) -> None:
        """
        SHAP summary (beeswarm) plot across many detections.
        Shows overall feature importance distribution.
        """
        sv = self.explainer.shap_values(
            feature_matrix.astype(np.float32),
            nsamples=SHAP_MAX_EVALS,
            silent=True
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            sv, feature_matrix,
            feature_names=self.feature_names,
            show=False, plot_type="dot",
            color_bar_label="Feature value"
        )
        plt.title("SHAP summary — feature importance across all detections", fontsize=11)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
