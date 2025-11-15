# app.py
import os
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
import numpy as np
import traceback

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- Helper: safe load resources ----------
def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Gagal load model {path}: {e}")
        traceback.print_exc()
        return None

def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Gagal load CSV {path}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def safe_load_geojson(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Gagal load GeoJSON {path}: {e}")
        traceback.print_exc()
        return {"type": "FeatureCollection", "features": []}

# ---------- Paths (pastikan ada di repo) ----------
MODEL_PATH = os.path.join("models", "model_stok_ikan.joblib")
CSV_PATH = os.path.join("data", "data_hasil_klasifikasi.csv")
GEOJSON_PATH = os.path.join("data", "provinsiIndonesia.json")

# ---------- Load resources ----------
model = safe_load_model(MODEL_PATH)
df = safe_load_csv(CSV_PATH)
geojson_data = safe_load_geojson(GEOJSON_PATH)

# If df has unexpected dtypes, try to coerce common columns
if not df.empty:
    # strip column names
    df.columns = df.columns.str.strip()
    # ensure Provinsi column exists
    if "Provinsi" in df.columns:
        df["Provinsi"] = df["Provinsi"].astype(str).str.upper()
    # ensure Tahun as int if possible
    if "Tahun" in df.columns:
        try:
            df["Tahun"] = df["Tahun"].astype(int)
        except Exception:
            pass

# Features expected by the model (keep as reference)
fitur_input_model = [
    "Effort (kapal)",
    "CPUE (Ton/Trip)",
    "Hasil Tangkapan / Catch (Ton)",
    "TP_C",
    "TP_E",
    "Tahun",
]

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    tahun_list = sorted(df["Tahun"].dropna().unique()) if not df.empty and "Tahun" in df.columns else []
    provinsi_list = sorted(df["Provinsi"].dropna().unique()) if not df.empty and "Provinsi" in df.columns else []
    ikan_list = sorted(df["Kelompok Ikan"].dropna().unique()) if not df.empty and "Kelompok Ikan" in df.columns else []
    return render_template(
        "dashboard.html",
        tahun_list=tahun_list,
        provinsi_list=provinsi_list,
        ikan_list=ikan_list,
    )

@app.route("/api/dashboard-populasi")
def api_dashboard_populasi():
    tahun = request.args.get("tahun")
    provinsi = request.args.get("provinsi")
    ikan = request.args.get("ikan")

    if df.empty:
        return jsonify([])

    df_dashboard = df.copy()
    # safe compute Populasi if columns exist
    if {"TP_C", "TP_E"}.issubset(df_dashboard.columns):
        df_dashboard["Populasi"] = df_dashboard[["TP_C", "TP_E"]].mean(axis=1)
    else:
        df_dashboard["Populasi"] = 0

    if tahun and tahun.strip():
        try:
            df_dashboard = df_dashboard[df_dashboard["Tahun"] == int(tahun)]
        except Exception:
            pass
    if provinsi and provinsi.strip():
        df_dashboard = df_dashboard[df_dashboard["Provinsi"] == provinsi.upper()]
    if ikan and ikan.strip():
        df_dashboard = df_dashboard[df_dashboard["Kelompok Ikan"] == ikan]

    cols = [c for c in ["Tahun", "Kelompok Ikan", "Provinsi", "Populasi"] if c in df_dashboard.columns]
    result = (
        df_dashboard[cols]
        .rename(columns={"Tahun": "tahun", "Kelompok Ikan": "kelompok_ikan", "Provinsi": "provinsi", "Populasi": "populasi"})
        .to_dict(orient="records")
    )
    return jsonify(result)

@app.route("/api/status-ikan")
def api_status_ikan():
    if df.empty:
        return jsonify([])

    kolom_yang_dipakai = ["Tahun", "Kelompok Ikan", "Provinsi", "MSY", "TP", "Status"]
    kolom_ada = [k for k in kolom_yang_dipakai if k in df.columns]
    data_filtered = df[kolom_ada].fillna("")
    return jsonify(data_filtered.to_dict(orient="records"))

@app.route("/api/card-infoekologi")
def api_card_infoekologi():
    if df.empty:
        return jsonify([])

    try:
        latest_year = int(df["Tahun"].max())
    except Exception:
        latest_year = None

    if latest_year is None:
        return jsonify([])

    df_latest = df[df["Tahun"] == latest_year]
    df_prev = df[df["Tahun"] == (latest_year - 1)]

    result = []
    for ikan in df_latest["Kelompok Ikan"].dropna().unique()[:5]:
        pop_now_c = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_C"].mean() if "TP_C" in df_latest.columns else 0
        pop_now_e = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_E"].mean() if "TP_E" in df_latest.columns else 0
        pop_now = (pop_now_c + pop_now_e) / 2

        pop_prev_c = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_C"].mean() if "TP_C" in df_prev.columns else 0
        pop_prev_e = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_E"].mean() if "TP_E" in df_prev.columns else 0
        pop_prev = (pop_prev_c + pop_prev_e) / 2

        trend = ((pop_now - pop_prev) / pop_prev) * 100 if pop_prev else 0
        status = df_latest[df_latest["Kelompok Ikan"] == ikan]["Status"].values[0] if "Status" in df_latest.columns else ""

        result.append({"nama": ikan, "populasi": round(pop_now, 2), "tren": round(trend, 2), "status": status})

    return jsonify(result)

# Helper: detect province property key in GeoJSON features
def detect_prov_property_key(features):
    candidates = {}
    for feat in features:
        props = feat.get("properties", {})
        for k in props.keys():
            k_low = k.lower()
            if any(token in k_low for token in ("prov", "provinsi", "name", "nama")):
                candidates[k] = candidates.get(k, 0) + 1
    if not candidates:
        return None
    # return the most common candidate
    return max(candidates, key=candidates.get)

@app.route("/api/peta-kepatuhan")
def api_peta_kepatuhan():
    tahun = request.args.get("tahun")
    provinsi = request.args.get("provinsi")
    ikan = request.args.get("ikan")

    if df.empty or not geojson_data.get("features"):
        return jsonify({"html": "<p>Data peta atau dataset tidak tersedia.</p>"})

    df_filtered = df.copy()
    # ensure uppercase provinsi column
    if "Provinsi" in df_filtered.columns:
        df_filtered["Provinsi"] = df_filtered["Provinsi"].astype(str).str.upper()

    # Filters
    if tahun and tahun.strip():
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(tahun)]
        except Exception:
            pass
    if provinsi and provinsi.strip():
        df_filtered = df_filtered[df_filtered["Provinsi"] == provinsi.upper()]
    if ikan and ikan.strip():
        df_filtered = df_filtered[df_filtered["Kelompok Ikan"] == ikan]

    if df_filtered.empty:
        return jsonify({"html": "<p>Tidak ada data untuk filter ini.</p>"})

    status_colors = {
        "UNDERFISHING": "#2ecc71",
        "UNCERTAIN": "#95a5a6",
        "DATA DEFICIENT": "#95a5a6",
        "OVERFISHING": "#e74c3c",
        "GROWTH OVERFISHING": "#f1c40f",
        "RECRUITMENT OVERFISHING": "#e67e22",
    }

    # summary text per provinsi
    def build_info_text(subdf):
        rows = []
        for _, r in subdf.iterrows():
            catch_col = "Hasil Tangkapan / Catch (Ton)" if "Hasil Tangkapan / Catch (Ton)" in subdf.columns else ("Catch" if "Catch" in subdf.columns else None)
            catch_val = r.get("Hasil Tangkapan / Catch (Ton)", r.get("Catch", 0)) if hasattr(r, "get") else r["Hasil Tangkapan / Catch (Ton)"]
            rows.append(f"{r['Kelompok Ikan']}: {catch_val:,} ton ({r.get('Status','')})")
        total = subdf["Hasil Tangkapan / Catch (Ton)"].sum() if "Hasil Tangkapan / Catch (Ton)" in subdf.columns else 0
        return "<br>".join(rows) + f"<br><br><b>Total tangkapan: {total:,.0f} ton</b>"

    df_summary = df_filtered.groupby("Provinsi").apply(build_info_text).to_dict()
    prov_status = df_filtered.groupby("Provinsi")["Status"].first().to_dict()

    # Detect which property key in GeoJSON contains province name
    prop_key = detect_prov_property_key(geojson_data["features"]) or "Provinsi"

    # Attach properties to GeoJSON features
    for feature in geojson_data["features"]:
        props = feature.setdefault("properties", {})
        prov_name = str(props.get(prop_key, props.get("Provinsi", ""))).upper()
        status = prov_status.get(prov_name, "Tidak ada Data")
        info = df_summary.get(prov_name, "Tidak ada data ikan untuk filter ini.")
        warna = status_colors.get(status.upper(), "#dcdcdc")

        props["Provinsi"] = prov_name
        props["Status"] = status
        props["info_ikan"] = info
        props["warna"] = warna

    # Build folium map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": feature["properties"].get("warna", "#dcdcdc"),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        tooltip=GeoJsonTooltip(fields=["Provinsi", "Status"], aliases=["Provinsi", "Status"], sticky=True),
        popup=GeoJsonPopup(fields=["Provinsi", "Status", "info_ikan"], aliases=["Provinsi:", "Status:", "Data Jenis Ikan:"], localize=True, labels=True),
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; width: 230px; height: 160px; 
    border:2px solid grey; z-index:9999; font-size:14px;
    background-color:white; padding: 10px;">
    <b>Status Pemanfaatan Ikan</b><br>
    <span style="color:#2ecc71;">●</span> Underfishing<br>
    <span style="color:#95a5a6;">●</span> Uncertain / Data Deficient<br>
    <span style="color:#e74c3c;">●</span> Overfishing<br>
    <span style="color:#f1c40f;">●</span> Growth Overfishing<br>
    <span style="color:#e67e22;">●</span> Recruitment Overfishing
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return jsonify({"html": m._repr_html_()})

@app.route("/marine-law")
def marine_law():
    return render_template("marine_law.html")

@app.route("/ecology-population")
def ecology_population():
    tahun_list = sorted(df["Tahun"].dropna().unique()) if not df.empty and "Tahun" in df.columns else []
    ikan_list = sorted(df["Kelompok Ikan"].dropna().unique()) if not df.empty and "Kelompok Ikan" in df.columns else []
    return render_template("ecology_population.html", tahun_list=tahun_list, ikan_list=ikan_list)

@app.route("/api/predict-overfishing", methods=["POST"])
def predict_overfishing():
    try:
        data = request.get_json(force=True)
        tahun = data.get("tahun")
        provinsi = data.get("provinsi")
        kelompok_ikan = data.get("kelompok_ikan")
        effort = float(data.get("effort", 0))
        cpue = float(data.get("cpue", 0))
        hasil_tangkapan = float(data.get("catch", 0))
        tp_c = float(data.get("tp_c", 0))
        tp_e = float(data.get("tp_e", 0))

        input_data = pd.DataFrame(
            [
                {
                    "Tahun": int(tahun),
                    "Provinsi": provinsi,
                    "Kelompok Ikan": kelompok_ikan,
                    "Effort (kapal)": effort,
                    "CPUE (Ton/Trip)": cpue,
                    "Hasil Tangkapan / Catch (Ton)": hasil_tangkapan,
                    "TP_C": tp_c,
                    "TP_E": tp_e,
                }
            ]
        )

        print("DataFrame untuk prediksi:\n", input_data)

        if model is None:
            return jsonify({"status": "error", "message": "Model tidak tersedia di server."}), 500

        pred = model.predict(input_data)
        result = str(pred[0])

        return jsonify({"status": "success", "prediction": result, "tahun": tahun, "provinsi": provinsi, "kelompok_ikan": kelompok_ikan})
    except Exception as e:
        print("Error saat prediksi:", str(e))
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug mode off by default in production (Gunicorn will run the app)
    app.run(host="0.0.0.0", port=port)

