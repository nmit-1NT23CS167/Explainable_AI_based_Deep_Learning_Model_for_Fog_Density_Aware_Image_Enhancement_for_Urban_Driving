"""SHAP KernelExplainer for fog alert risk scoring."""
import os, numpy as np
import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import SHAP_MAX_EVALS, SHAP_TOP_K_FEATURES, FEATURE_NAMES, OUTPUT_DIR
from utils.distance_estimator import ZONE_CAUTION


def compute_risk_score(features: np.ndarray) -> np.ndarray:
    features = np.atleast_2d(features).astype(np.float32)
    risks = np.zeros(len(features))
    for i, f in enumerate(features):
        bw, bh = max(f[0],1.), max(f[1],1.)
        cx, cy, conf, fog = f[3], f[4], np.clip(f[5],0,1), np.clip(f[6],0,1)
        dist  = max(1., 1440./bw)
        d_risk = np.clip(1. - dist/ZONE_CAUTION, 0, 1)
        f_risk = np.clip(fog*(1+d_risk), 0, 1)
        p_risk = np.clip(1. - 2*abs(cx-0.5), 0, 1)
        risks[i] = np.clip(.45*d_risk + .30*f_risk + .15*p_risk + .10*conf*.2, 0, 1)
    return risks


class FogAlertSHAPExplainer:
    def __init__(self, background: np.ndarray):
        self.explainer = shap.KernelExplainer(
            compute_risk_score, background, link="identity")
        self.feature_names = FEATURE_NAMES

    def explain(self, fv: np.ndarray) -> dict:
        x  = np.atleast_2d(fv.astype(np.float32))
        sv = np.array(self.explainer.shap_values(
            x, nsamples=SHAP_MAX_EVALS, silent=True)).flatten()
        risk = float(compute_risk_score(x)[0])
        base = float(self.explainer.expected_value)
        ranked = sorted(zip(self.feature_names, sv),
                        key=lambda p: abs(p[1]), reverse=True)
        return {"shap_values": sv, "base_value": base,
                "risk_score": risk, "top_features": ranked[:SHAP_TOP_K_FEATURES]}

    def plot_waterfall(self, result, title="", save_path=None):
        names  = [f[0] for f in result["top_features"]]
        values = [f[1] for f in result["top_features"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in values]
        bars = ax.barh(names, values, color=colors, height=0.55)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        for bar, val in zip(bars, values):
            sign = "+" if val >= 0 else ""
            ax.text(val+(0.003 if val>=0 else -0.003),
                    bar.get_y()+bar.get_height()/2,
                    f"{sign}{val:.3f}", va="center",
                    ha="left" if val>=0 else "right", fontsize=9)
        ax.set_xlabel("SHAP value (contribution to risk score)")
        ax.set_title(f"{title}\nBase: {result['base_value']:.3f}  "
                     f"→  Risk: {result['risk_score']:.3f}", fontsize=10)
        ax.set_xlim(-0.6, 0.6)
        ax.legend(handles=[Patch(color="#d73027", label="Increases risk"),
                            Patch(color="#4575b4", label="Decreases risk")],
                  loc="lower right", fontsize=8)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def plot_summary(self, feature_matrix, save_path=None):
        sv = self.explainer.shap_values(feature_matrix.astype(np.float32),
                                        nsamples=SHAP_MAX_EVALS, silent=True)
        fig, _ = plt.subplots(figsize=(9, 5))
        shap.summary_plot(sv, feature_matrix,
                          feature_names=self.feature_names,
                          show=False, plot_type="dot")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
