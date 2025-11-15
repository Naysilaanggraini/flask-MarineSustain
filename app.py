from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib 
import geopandas as gpd
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
import json
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("models/model_stok_ikan.joblib")
df = pd.read_csv("data/data_hasil_klasifikasi.csv")
geojson_path = "data/provinsiIndonesia.json"
gdf = gpd.read_file(geojson_path)
fitur_input_model = ['Effort (kapal)', 'CPUE (Ton/Trip)', 'Hasil Tangkapan / Catch (Ton)', 'TP_C', 'TP_E', 'Tahun']

# ROUTES / HALAMAN 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    tahun_list = sorted(df['Tahun'].unique())
    provinsi_list = sorted(df['Provinsi'].unique())
    ikan_list = sorted(df['Kelompok Ikan'].unique())
    return render_template('dashboard.html',
                           tahun_list=tahun_list,
                           provinsi_list=provinsi_list,
                           ikan_list=ikan_list)

# ==========================
# === API DASHBOARD DATA ===
# ==========================

@app.route('/api/dashboard-populasi')
def api_dashboard_populasi():
    tahun = request.args.get('tahun')
    provinsi = request.args.get('provinsi')
    ikan = request.args.get('ikan')

    df_dashboard = df.copy()
    df_dashboard["Populasi"] = df_dashboard[["TP_C", "TP_E"]].mean(axis=1)

    if tahun and tahun.strip():
        df_dashboard = df_dashboard[df_dashboard["Tahun"] == int(tahun)]
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


 # === space api buat grafik ekologi === #
 # APIEKOLOGIPOPULATION
@app.route('/api/status-ikan')
def api_status_ikan():
    if df.empty:
        return jsonify([])

    # memilih kolom utama
    kolom_yang_dipakai = ['Tahun', 'Kelompok Ikan', 'Provinsi', 'MSY', 'TP', 'Status']
    kolom_ada = [k for k in kolom_yang_dipakai if k in df.columns]
    data_filtered = df[kolom_ada]

    data_filtered = data_filtered.fillna('')

    return jsonify(data_filtered.to_dict(orient='records'))

@app.route("/api/card-infoekologi")
def api_card_infoekologi():
    if df.empty:
        return jsonify([])

    # mengambil data terbaru (tahun 2024)
    latest_year = df["Tahun"].max()
    df_latest = df[df["Tahun"] == latest_year]

    # Hitung tren dari tahun sebelumnya
    prev_year = latest_year - 1
    df_prev = df[df["Tahun"] == prev_year]

    result = []
    for ikan in df_latest["Kelompok Ikan"].unique()[:5]:  # 5 jenis ikan
        pop_now_c = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_C"].mean()
        pop_now_e = df_latest[df_latest["Kelompok Ikan"] == ikan]["TP_E"].mean()
        pop_now = (pop_now_c + pop_now_e) / 2

        pop_prev_c = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_C"].mean()
        pop_prev_e = df_prev[df_prev["Kelompok Ikan"] == ikan]["TP_E"].mean()
        pop_prev = (pop_prev_c + pop_prev_e) / 2

        trend = ((pop_now - pop_prev) / pop_prev) * 100 if pop_prev else 0
        status = df_latest[df_latest["Kelompok Ikan"] == ikan]["Status"].values[0]

        result.append({
            "nama": ikan,
            "populasi": round(pop_now, 2),
            "tren": round(trend, 2),
            "status": status
        })

    return jsonify(result)


@app.route("/api/peta-kepatuhan")
def api_peta_kepatuhan():
    tahun = request.args.get("tahun")
    provinsi = request.args.get("provinsi")
    ikan = request.args.get("ikan")

    df_filtered = df.copy()

    # === Normalisasi kolom ===
    df_filtered.columns = df_filtered.columns.str.strip()
    gdf.columns = gdf.columns.str.strip()

    print("Kolom di shapefile:", list(gdf.columns))

    # === Deteksi otomatis kolom provinsi di shapefile ===
    prov_col_candidates = [c for c in gdf.columns if "provinsi" in c.lower() or "wadmpr" in c.lower()]
    if not prov_col_candidates:
    # fallback kalau shapefile pakai nama lain
        prov_col_candidates = [c for c in gdf.columns if "prov" in c.lower()]
    if prov_col_candidates:
        prov_col = prov_col_candidates[0]
        gdf["Provinsi"] = gdf[prov_col].astype(str).str.upper()
        print(f"Kolom provinsi terdeteksi otomatis: {prov_col}")
    else:
        return jsonify({"error": "Kolom provinsi tidak ditemukan di shapefile"}), 400


    # Samakan format provinsi di CSV
    df_filtered["Provinsi"] = df_filtered["Provinsi"].str.upper()

    # === Filter data ===
    if tahun and tahun.strip():
        df_filtered = df_filtered[df_filtered["Tahun"] == int(tahun)]
    if provinsi and provinsi.strip():
        df_filtered = df_filtered[df_filtered["Provinsi"] == provinsi.upper()]
    if ikan and ikan.strip():
        df_filtered = df_filtered[df_filtered["Kelompok Ikan"] == ikan]

    if df_filtered.empty:
        return jsonify({"html": "<p>Tidak ada data untuk filter ini.</p>"})

    # === Warna status ===
    status_colors = {
        "UNDERFISHING": "#2ecc71",
        "UNCERTAIN": "#95a5a6",
        "DATA DEFICIENT": "#95a5a6",
        "OVERFISHING": "#e74c3c",
        "GROWTH OVERFISHING": "#f1c40f",
        "RECRUITMENT OVERFISHING": "#e67e22"
    }

    # === Ringkasan per provinsi ===
    def build_info_text(subdf):
        rows = []
        for _, r in subdf.iterrows():
            rows.append(f"{r['Kelompok Ikan']}: {r['Hasil Tangkapan / Catch (Ton)']:,} ton ({r['Status']})")
        total = subdf["Hasil Tangkapan / Catch (Ton)"].sum()
        return "<br>".join(rows) + f"<br><br><b>Total tangkapan: {total:,.0f} ton</b>"

    df_summary = (
        df_filtered.groupby("Provinsi")
        .apply(build_info_text)
        .reset_index(name="info_ikan")
    )

    # === Merge dengan shapefile ===
    merged = (
        gdf.merge(df_filtered.groupby("Provinsi")["Status"].first().reset_index(), on="Provinsi", how="left")
           .merge(df_summary, on="Provinsi", how="left")
    )

    merged["warna"] = merged["Status"].str.upper().map(status_colors).fillna("#dcdcdc")
    merged["info_ikan"] = merged["info_ikan"].fillna("Tidak ada data ikan untuk filter ini.")
    merged["Status"] = merged["Status"].fillna("Tidak ada Data")

    # === Buat peta ===
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    folium.GeoJson(
        merged,
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

    # === Legend ===
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
    
@app.route('/marine-law')
def marine_law():
    return render_template('marine_law.html')

@app.route('/ecology-population')
def ecology_population():
    # ambil daftar tahun & jenis ikan unik buat dropdown/filter (opsional)
    tahun_list = sorted(df['Tahun'].unique())
    ikan_list = sorted(df['Kelompok Ikan'].unique())
    return render_template('ecology_population.html',
                           tahun_list=tahun_list,
                           ikan_list=ikan_list)

@app.route('/api/predict-overfishing', methods=['POST'])
def predict_overfishing():
    try:
        data = request.get_json()
        print("Data diterima:", data)

        # Ambil nilai dari form
        tahun = data.get("tahun")
        provinsi = data.get("provinsi")
        kelompok_ikan = data.get("kelompok_ikan")
        effort = float(data.get("effort", 0))
        cpue = float(data.get("cpue", 0))
        hasil_tangkapan = float(data.get("catch", 0))
        tp_c = float(data.get("tp_c", 0))
        tp_e = float(data.get("tp_e", 0))

        # Buat DataFrame SESUAI DENGAN FITUR MODEL
        input_data = pd.DataFrame([{
            "Tahun": int(tahun),
            "Provinsi": provinsi,
            "Kelompok Ikan": kelompok_ikan,
            "Effort (kapal)": effort,
            "CPUE (Ton/Trip)": cpue,
            "Hasil Tangkapan / Catch (Ton)": hasil_tangkapan,
            "TP_C": tp_c,
            "TP_E": tp_e
        }])

        print("\nDataFrame untuk prediksi:\n", input_data)

        # === Lakukan Prediksi ===
        pred = model.predict(input_data)
        result = str(pred[0])

        print("Hasil prediksi:", result)

        return jsonify({
            "status": "success",
            "prediction": result,
            "tahun": tahun,
            "provinsi": provinsi,
            "kelompok_ikan": kelompok_ikan
        })

    except Exception as e:
        print("Error saat prediksi:", str(e))
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


