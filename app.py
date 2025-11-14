from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
import json
import numpy as np
import os

app = Flask(__name__)

# ========== LOAD MODEL & DATA ==========
# Joblib model
MODEL_PATH = "models/model_stok_ikan.joblib"
CSV_PATH = "data/data_hasil_klasifikasi.csv"
GEOJSON_PATH = "data/provinsiIndonesia.json"

# load model & csv at startup (ok if file sizes reasonable)
model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

# normalize column names (strip)
df.columns = df.columns.str.strip()

# fitur model (untuk referensi)
fitur_input_model = [
    'Effort (kapal)', 'CPUE (Ton/Trip)', 'Hasil Tangkapan / Catch (Ton)',
    'TP_C', 'TP_E', 'Tahun'
]

# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    tahun_list = sorted(df['Tahun'].dropna().unique().tolist())
    provinsi_list = sorted(df['Provinsi'].dropna().unique().tolist())
    ikan_list = sorted(df['Kelompok Ikan'].dropna().unique().tolist())
    return render_template('dashboard.html',
                           tahun_list=tahun_list,
                           provinsi_list=provinsi_list,
                           ikan_list=ikan_list)

# =========================
# API: dashboard-populasi
# =========================
@app.route('/api/dashboard-populasi')
def api_dashboard_populasi():
    tahun = request.args.get('tahun')
    provinsi = request.args.get('provinsi')
    ikan = request.args.get('ikan')

    df_dashboard = df.copy()
    # safety: ensure TP_C & TP_E exist
    if 'TP_C' in df_dashboard.columns and 'TP_E' in df_dashboard.columns:
        df_dashboard["Populasi"] = df_dashboard[["TP_C", "TP_E"]].mean(axis=1)
    else:
        df_dashboard["Populasi"] = np.nan

    if tahun and tahun.strip():
        try:
            df_dashboard = df_dashboard[df_dashboard["Tahun"] == int(tahun)]
        except Exception:
            pass
    if provinsi and provinsi.strip():
        df_dashboard = df_dashboard[df_dashboard["Provinsi"] == provinsi]
    if ikan and ikan.strip():
        df_dashboard = df_dashboard[df_dashboard["Kelompok Ikan"] == ikan]

    result = (
        df_dashboard[["Tahun", "Kelompok Ikan", "Provinsi", "Populasi"]]
        .rename(columns={
            "Tahun": "tahun",
            "Kelompok Ikan": "kelompok_ikan",
            "Provinsi": "provinsi",
            "Populasi": "populasi"
        })
        .to_dict(orient="records")
    )
    return jsonify(result)

# =========================
# API: status-ikan
# =========================
@app.route('/api/status-ikan')
def api_status_ikan():
    if df.empty:
        return jsonify([])

    kolom_yang_dipakai = ['Tahun', 'Kelompok Ikan', 'Provinsi', 'MSY', 'TP', 'Status']
    kolom_ada = [k for k in kolom_yang_dipakai if k in df.columns]
    data_filtered = df[kolom_ada].fillna('')
    return jsonify(data_filtered.to_dict(orient='records'))

# =========================
# API: card-infoekologi
# =========================
@app.route("/api/card-infoekologi")
def api_card_infoekologi():
    if df.empty:
        return jsonify([])

    latest_year = df["Tahun"].max()
    df_latest = df[df["Tahun"] == latest_year]
    prev_year = latest_year - 1
    df_prev = df[df["Tahun"] == prev_year]

    result = []
    for ikan in df_latest["Kelompok Ikan"].unique()[:5]:
        pop_now_c = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_C"].mean()
        pop_now_e = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_E"].mean()
        pop_now = (pop_now_c + pop_now_e) / 2 if not np.isnan(pop_now_c) or not np.isnan(pop_now_e) else 0

        pop_prev_c = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_C"].mean()
        pop_prev_e = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_E"].mean()
        pop_prev = (pop_prev_c + pop_prev_e) / 2 if not np.isnan(pop_prev_c) or not np.isnan(pop_prev_e) else 0

        trend = ((pop_now - pop_prev) / pop_prev) * 100 if pop_prev else 0
        status = df_latest[df_latest["Kelompok Ikan"] == ikan]["Status"].values[0] if not df_latest[df_latest["Kelompok Ikan"] == ikan].empty else "Tidak Ada Data"

        result.append({
            "nama": ikan,
            "populasi": round(float(pop_now), 2),
            "tren": round(float(trend), 2),
            "status": status
        })

    return jsonify(result)

# =========================
# API: peta-kepatuhan (TANPA GeoPandas)
# =========================
@app.route("/api/peta-kepatuhan")
def api_peta_kepatuhan():
    # params
    tahun = request.args.get("tahun")
    provinsi = request.args.get("provinsi")
    ikan = request.args.get("ikan")

    # copy dataframe and normalize Provinsi column
    df_filtered = df.copy()
    if "Provinsi" in df_filtered.columns:
        df_filtered["Provinsi_norm"] = df_filtered["Provinsi"].astype(str).str.strip().str.upper()
    else:
        df_filtered["Provinsi_norm"] = ""

    # apply filters
    if tahun and tahun.strip():
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(tahun)]
        except Exception:
            pass
    if provinsi and provinsi.strip():
        df_filtered = df_filtered[df_filtered["Provinsi_norm"] == provinsi.strip().upper()]
    if ikan and ikan.strip():
        df_filtered = df_filtered[df_filtered["Kelompok Ikan"] == ikan]

    if df_filtered.empty:
        return jsonify({"html": "<p>Tidak ada data untuk filter ini.</p>"})

    # build summary per provinsi
    def build_info_text(subdf):
        rows = []
        for _, r in subdf.iterrows():
            catch = r.get("Hasil Tangkapan / Catch (Ton)", 0)
            try:
                catch_fmt = f"{int(catch):,}"
            except Exception:
                catch_fmt = str(catch)
            rows.append(f"{r.get('Kelompok Ikan','')}: {catch_fmt} ton ({r.get('Status','')})")
        total = subdf["Hasil Tangkapan / Catch (Ton)"].sum() if "Hasil Tangkapan / Catch (Ton)" in subdf.columns else 0
        try:
            total_fmt = f"{int(total):,}"
        except Exception:
            total_fmt = str(total)
        return "<br>".join(rows) + f"<br><br><b>Total tangkapan: {total_fmt} ton</b>"

    df_summary = df_filtered.groupby("Provinsi").apply(build_info_text).reset_index(name="info_ikan")
    df_status = df_filtered.groupby("Provinsi")["Status"].first().reset_index()

    # load geojson as dict
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except Exception as e:
        return jsonify({"html": f"<p>Gagal memuat file geojson: {str(e)}</p>"})

    # prepare mapping dicts: uppercase province name -> status/info
    summary_map = {r["Provinsi"].strip().upper(): r["info_ikan"] for _, r in df_summary.iterrows()}
    status_map = {r["Provinsi"].strip().upper(): r["Status"] for _, r in df_status.iterrows()}

    # warna untuk status (kecilkan key ke upper)
    status_colors = {
        "UNDERFISHING": "#2ecc71",
        "UNCERTAIN": "#95a5a6",
        "DATA DEFICIENT": "#95a5a6",
        "OVERFISHING": "#e74c3c",
        "GROWTH OVERFISHING": "#f1c40f",
        "RECRUITMENT OVERFISHING": "#e67e22"
    }

    # iterate features and inject properties: Provinsi (normalized), Status, info_ikan, warna
    features = geojson.get("features", [])
    for feat in features:
        props = feat.get("properties", {}) or {}
        # try several candidate keys to find province name in geojson properties
        prov_name = ""
        for candidate in ["provinsi", "Provinsi", "NAME_1", "name", "WADMPR", "WADMPRINS"]:
            if candidate in props and props[candidate]:
                prov_name = str(props[candidate]).strip()
                break
        if not prov_name:
            # fallback: try first string property
            for k, v in props.items():
                if isinstance(v, str) and v.strip():
                    prov_name = v.strip()
                    break

        prov_norm = prov_name.upper()
        # attach properties that folium tooltip/popup expect
        props["Provinsi"] = prov_name
        props["Status"] = status_map.get(prov_norm, "Tidak ada Data")
        props["info_ikan"] = summary_map.get(prov_norm, "Tidak ada data ikan untuk filter ini.")
        props["warna"] = status_colors.get(str(props["Status"]).strip().upper(), "#dcdcdc")

        feat["properties"] = props

    # build folium map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            "fillColor": feature["properties"].get("warna", "#dcdcdc"),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        tooltip=GeoJsonTooltip(fields=["Provinsi", "Status"], aliases=["Provinsi", "Status"], sticky=True),
        popup=GeoJsonPopup(
            fields=["Provinsi", "Status", "info_ikan"],
            aliases=["Provinsi:", "Status:", "Data Jenis Ikan:"],
            localize=True,
            labels=True,
            style="background-color: white; border-radius: 5px; padding: 5px;"
        )
    ).add_to(m)

    # legend (static html)
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

    # return folium map html embedded in JSON — frontend JS expects {"html": "..."}
    return jsonify({"html": m._repr_html_()})

# =========================
# API: predict-overfishing
# =========================
@app.route('/api/predict-overfishing', methods=['POST'])
def predict_overfishing():
    try:
        data = request.get_json(force=True)
        # build input dataframe for model
        input_data = pd.DataFrame([{
            "Tahun": int(data.get("tahun", 0)),
            "Provinsi": data.get("provinsi", ""),
            "Kelompok Ikan": data.get("kelompok_ikan", ""),
            "Effort (kapal)": float(data.get("effort", 0)),
            "CPUE (Ton/Trip)": float(data.get("cpue", 0)),
            "Hasil Tangkapan / Catch (Ton)": float(data.get("catch", 0)),
            "TP_C": float(data.get("tp_c", 0)),
            "TP_E": float(data.get("tp_e", 0))
        }])

        pred = model.predict(input_data)
        result = str(pred[0])

        return jsonify({
            "status": "success",
            "prediction": result,
            "tahun": data.get("tahun"),
            "provinsi": data.get("provinsi"),
            "kelompok_ikan": data.get("kelompok_ikan")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ========== RUN ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug False for production; Railway handles logs
    app.run(host="0.0.0.0", port=port)
