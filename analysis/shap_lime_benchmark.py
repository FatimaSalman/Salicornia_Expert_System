"""
SHAP/LIME benchmarking against the rule-based Inference Engine
for the Salicornia europaea Hybrid Expert System (post-COMPAG revision).

Reproduces the manuscript's RF models (salinity -> biomass / stiffness),
verifies LOO-CV metrics (Table 1), then applies SHAP (TreeExplainer) and
LIME (tabular) to the three manuscript scenarios (0, 400, 1000 mM NaCl)
and contrasts their outputs with the Inference Engine's biochemical
explanations derived from the Pearson correlation matrix (Table S2 of [1]).
"""
import io, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from lime.lime_tabular import LimeTabularExplainer

RNG = 42
np.random.seed(RNG)

# ----- Empirical dataset (Ciechocinek, n = 6) -----
salinity = np.array([0, 200, 400, 600, 800, 1000], dtype=float)
biomass  = np.array([3.74, 9.73, 10.10, 6.47, 1.09, 0.50])
stiff    = np.array([1.790, 1.278, 1.304, 1.003, 0.605, 0.357])
X = salinity.reshape(-1, 1)

# ----- Pearson correlation knowledge base (Table S2 of [1]) -----
corr_csv = """Component,E stiffness,FW
Pectin HM-HG,0.556,0.897
Cellulose,0.950,0.519
S/G,0.239,0.944
H/G,-0.002,-0.813
S,-0.080,0.818
G,-0.602,0.410
H,-0.889,-0.516
Lignin-Total yield,-0.349,0.646
"""
kb = pd.read_csv(io.StringIO(corr_csv), index_col=0)

def inference_engine(target):
    """Mimics the manuscript's rule-based engine: returns strongest correlates."""
    col = "E stiffness" if target == "stiffness" else "FW"
    s = kb[col].reindex(kb[col].abs().sort_values(ascending=False).index)
    top = s.head(2)
    return [(name, float(r)) for name, r in top.items()]

# ----- 1. Reproduce LOO-CV Table 1 -----
def loo_eval(model_factory, X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return (r2_score(y, preds),
            float(np.sqrt(mean_squared_error(y, preds))),
            float(mean_absolute_error(y, preds)))

factories = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=RNG),
    "Linear Regression": lambda: LinearRegression(),
    "SVR (RBF)": lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma=0.01)),
    "Polynomial (deg=2)": lambda: make_pipeline(PolynomialFeatures(2), LinearRegression()),
}
rows = []
for name, f in factories.items():
    for var, y in [("Stiffness (MPa)", stiff), ("Biomass (g)", biomass)]:
        r2, rmse, mae = loo_eval(f, X, y)
        rows.append({"Model": name, "Variable": var, "LOO-R2": round(r2, 4),
                     "LOO-RMSE": round(rmse, 4), "LOO-MAE": round(mae, 4)})
table1 = pd.DataFrame(rows)
print("=== Reproduced LOO-CV (compare with manuscript Table 1) ===")
print(table1.to_string(index=False))
table1.to_csv("loocv_reproduced.csv", index=False)

# ----- 2. Fit final RF models on all 6 points (as deployed) -----
rf_b = RandomForestRegressor(n_estimators=100, random_state=RNG).fit(X, biomass)
rf_s = RandomForestRegressor(n_estimators=100, random_state=RNG).fit(X, stiff)

scenarios = {"B (freshwater)": 0.0, "A (optimal)": 400.0, "C (extreme)": 1000.0}

# ----- 3. SHAP -----
expl_b = shap.TreeExplainer(rf_b)
expl_s = shap.TreeExplainer(rf_s)

grid = np.linspace(0, 1000, 201).reshape(-1, 1)
shap_grid_b = expl_b.shap_values(grid).ravel()
shap_grid_s = expl_s.shap_values(grid).ravel()
base_b = float(np.ravel(expl_b.expected_value)[0])
base_s = float(np.ravel(expl_s.expected_value)[0])

shap_scen = {}
for label, mM in scenarios.items():
    xx = np.array([[mM]])
    shap_scen[label] = {
        "salinity_mM": mM,
        "biomass_pred": float(rf_b.predict(xx)[0]),
        "biomass_base": base_b,
        "biomass_shap_salinity": float(expl_b.shap_values(xx).ravel()[0]),
        "stiff_pred": float(rf_s.predict(xx)[0]),
        "stiff_base": base_s,
        "stiff_shap_salinity": float(expl_s.shap_values(xx).ravel()[0]),
    }

# ----- 4. LIME -----
lime_expl = LimeTabularExplainer(
    training_data=X, feature_names=["Salinity_mM"], mode="regression",
    discretize_continuous=False, random_state=RNG)

lime_scen = {}
for label, mM in scenarios.items():
    e_b = lime_expl.explain_instance(np.array([mM]), rf_b.predict, num_features=1)
    e_s = lime_expl.explain_instance(np.array([mM]), rf_s.predict, num_features=1)
    lime_scen[label] = {
        "biomass_lime_weight": float(e_b.as_list()[0][1]),
        "biomass_lime_intercept": float(e_b.intercept[0]) if hasattr(e_b.intercept, "__getitem__") else float(e_b.intercept),
        "stiff_lime_weight": float(e_s.as_list()[0][1]),
    }

# ----- 5. Side-by-side explanation comparison -----
print("\n=== Explanation comparison (per scenario) ===")
comp_rows = []
for label, mM in scenarios.items():
    s = shap_scen[label]; l = lime_scen[label]
    ie_s = inference_engine("stiffness"); ie_b = inference_engine("biomass")
    comp_rows.append({
        "Scenario": label, "Salinity (mM)": mM,
        "RF biomass pred (g)": round(s["biomass_pred"], 2),
        "SHAP(sal)->biomass": round(s["biomass_shap_salinity"], 3),
        "LIME w(sal)->biomass": round(l["biomass_lime_weight"], 5),
        "RF stiffness pred (MPa)": round(s["stiff_pred"], 3),
        "SHAP(sal)->stiffness": round(s["stiff_shap_salinity"], 3),
        "LIME w(sal)->stiffness": round(l["stiff_lime_weight"], 6),
        "Inference Engine (stiffness)": "; ".join(f"{n} (r={r:+.3f})" for n, r in ie_s),
        "Inference Engine (biomass)": "; ".join(f"{n} (r={r:+.3f})" for n, r in ie_b),
    })
comp = pd.DataFrame(comp_rows)
print(comp.to_string(index=False))
comp.to_csv("xai_benchmark_comparison.csv", index=False)

# Biochemical information content (count of biochemical terms in each explanation)
print("\nBiochemical constructs referenced: SHAP = 0, LIME = 0, Inference Engine = 2 per target "
      "(by construction: salinity is the model's only input feature).")

# ----- 6. Figures -----
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# Fig A: SHAP dependence across salinity gradient (both targets)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].axhline(0, color="grey", lw=0.8)
axes[0].plot(grid.ravel(), shap_grid_b, color="#2e7d32", lw=2)
axes[0].scatter(salinity, expl_b.shap_values(X).ravel(), color="#1b5e20", zorder=5,
                label="Training points")
axes[0].set_xlabel("Salinity (mM NaCl)"); axes[0].set_ylabel("SHAP value for salinity (g)")
axes[0].set_title(f"Biomass: SHAP attribution\n(base value = {base_b:.2f} g)")
axes[0].legend()
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].plot(grid.ravel(), shap_grid_s, color="#1565c0", lw=2)
axes[1].scatter(salinity, expl_s.shap_values(X).ravel(), color="#0d47a1", zorder=5,
                label="Training points")
axes[1].set_xlabel("Salinity (mM NaCl)"); axes[1].set_ylabel("SHAP value for salinity (MPa)")
axes[1].set_title(f"Stiffness: SHAP attribution\n(base value = {base_s:.3f} MPa)")
axes[1].legend()
fig.tight_layout()
fig.savefig("../figures/Fig_SHAP_dependence.png", dpi=300, bbox_inches="tight")

# Fig B: three-way explanation comparison for Scenario C (1000 mM)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axis("off")
sC = shap_scen["C (extreme)"]; lC = lime_scen["C (extreme)"]
txt = (
    "Scenario C - Extreme salinity (1000 mM NaCl), stiffness prediction = "
    f"{sC['stiff_pred']:.3f} MPa\n\n"
    f"SHAP (TreeExplainer):   f(x) = base {sC['stiff_base']:.3f} MPa "
    f"+ phi(Salinity) = {sC['stiff_shap_salinity']:+.3f} MPa\n"
    "   -> 100% of the deviation attributed to 'Salinity_mM' (the model's only feature).\n"
    "   -> No biochemical construct available to the explainer.\n\n"
    f"LIME (local linear):    w(Salinity) = {lC['stiff_lime_weight']:+.3f} MPa per SD of salinity\n"
    "   -> A local slope on the same single feature; again no biochemical content.\n\n"
    "Rule-based Inference Engine (knowledge base = Pearson matrix, Table S2 of [1]):\n"
    "   -> 'Marked loss of stiffness is expected; strongest empirical correlates:\n"
    "       Cellulose (r = +0.950) [reduced deposition], H lignin monomer (r = -0.889)\n"
    "       [H-rich lignin accumulation]' -> mechanistic, biologically testable hypothesis."
)
ax.text(0.01, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9.5)
ax.set_title("Three-way explanation comparison at 1000 mM NaCl", fontsize=12, pad=12)
fig.tight_layout()
fig.savefig("../figures/Fig_threeway_explanation.png", dpi=300, bbox_inches="tight")

json.dump({"shap": shap_scen, "lime": lime_scen,
           "inference_engine": {"stiffness": inference_engine("stiffness"),
                                "biomass": inference_engine("biomass")}},
          open("xai_results.json", "w"), indent=2)
print("\nSaved: loocv_reproduced.csv, xai_benchmark_comparison.csv, xai_results.json, 2 figures.")
