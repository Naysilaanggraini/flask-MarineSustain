document.addEventListener("DOMContentLoaded", async () => {
  try {
    // 🔹 Ambil data dari API Flask
    const res = await fetch("/api/status-ikan");
    const data = await res.json();

    if (!data || data.length === 0) {
      console.warn("Data kosong dari API");
      return;
    }

    // 🔹 Ambil daftar tahun unik (2021–2024)
    const tahunLabels = [...new Set(data.map(d => d.Tahun))].sort();

    // 🔹 Daftar kategori status (5 kategori tetap)
    const kategori = [
      "Underfishing",
      "Optimal",
      "Growth Overfishing",
      "Recruitment Overfishing",
      "Uncertain"
    ];

    // 🔹 Warna untuk tiap kategori
    const warna = {
      "Underfishing": "#3B82F6",           // biru
      "Optimal": "#22C55E",                // hijau
      "Growth Overfishing": "#FACC15",     // kuning
      "Recruitment Overfishing": "#EF4444",// merah
      "Uncertain": "#9CA3AF"               // abu
    };

    // 🔹 Hitung jumlah tiap kategori per tahun
    const countByYear = {};
    tahunLabels.forEach(t => {
      countByYear[t] = {};
      kategori.forEach(k => (countByYear[t][k] = 0));
    });

    data.forEach(d => {
      const t = d.Tahun;
      const s = d.Status?.trim();
      if (countByYear[t] && kategori.includes(s)) {
        countByYear[t][s] += 1;
      }
    });

    // 🔹 Dataset untuk Chart.js (grafik batang)
    const datasets = kategori.map(k => ({
      label: k,
      data: tahunLabels.map(t => countByYear[t][k]),
      backgroundColor: warna[k],
      borderWidth: 1
    }));

    // 🔹 Render grafik batang
    const ctx = document.getElementById("ikanChart").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: tahunLabels,
        datasets: datasets
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: false // sembunyikan legend default Chart.js
          },
          title: {
            display: true,
            text: "Distribusi Status Stok Ikan 2021–2024 (5 Kategori)",
            font: { size: 16, weight: "bold" }
          },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.formattedValue}`
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            title: { display: true, text: "Tahun" }
          },
          y: {
            stacked: true,
            beginAtZero: true,
            title: { display: true, text: "Jumlah Data" }
          }
        }
      }
    });

    // 🔹 Buat pie chart komposisi total semua tahun
    const totalStatus = kategori.map(k =>
      data.filter(d => d.Status === k).length
    );

    const pieCtx = document.getElementById("pieChart").getContext("2d");
    new Chart(pieCtx, {
      type: "doughnut",
      data: {
        labels: kategori,
        datasets: [{
          data: totalStatus,
          backgroundColor: kategori.map(k => warna[k])
        }]
      },
      options: {
        plugins: {
          title: { 
            display: true, 
            text: "Komposisi Status Stok Ikan (Total 2021–2024)",
            font: { size: 14, weight: "bold" }
          },
          legend: { 
            position: "right" 
          }
        }
      }
    });

     // === CARD INFO EKOLOGI ===
async function loadCardInfo() {
  try {
    const res = await fetch("/api/card-infoekologi");
    const data = await res.json();
    const container = document.getElementById("ikanCards");

    if (!data || data.length === 0) {
      container.innerHTML = "<p>Data tidak tersedia.</p>";
      return;
    }

    container.innerHTML = ""; // bersihkan dulu

    data.forEach((d, i) => {
      // Warna & emoji status
      let statusClass = "status success";
      let emoji = " ";
      if (d.status.toLowerCase().includes("over")) statusClass = "status warning";
      else if (d.status.toLowerCase().includes("uncertain")) statusClass = "status neutral";

      if (i === 1);
      if (i === 2);
      if (i === 3);
      if (i === 4);

      // Warna tren
      const trenColor = d.tren >= 0 ? "positive" : "warning";
      const trenSymbol = d.tren >= 0 ? "+" : "";

      // Tambahkan card ke halaman
      container.innerHTML += `
        <div class="card">
          <h3>${emoji} ${d.nama}</h3>
          <p>Populasi Saat Ini: <b>${d.populasi.toLocaleString()}</b></p>
          <p>Tren Tahunan: <span class="${trenColor}">${trenSymbol}${d.tren}%</span></p>
          <p>Status: <span class="${statusClass}">${d.status}</span></p>
        </div>
      `;
    });
  } catch (err) {
    console.error("Gagal memuat card info ekologi:", err);
  }
}

loadCardInfo();


  } catch (err) {
    console.error("Gagal memuat data:", err);
  }
});


// === FORM PREDIKSI OVERFISHING ===
document.getElementById("predictForm")?.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Ambil data dari form
  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries()); // ubah ke objek JS

  try {
    const res = await fetch("/api/predict-overfishing", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data), // kirim JSON
    });

    const result = await res.json();
    const resultBox = document.getElementById("result");

    if (result.status === "success") {
      const status = result.prediction;
      let color = "#6B7280";

      if (status.includes("OVER")) color = "#EF4444";
      else if (status.includes("UNDER")) color = "#22C55E";
      else if (status.includes("MOD")) color = "#FACC15";

      resultBox.innerHTML = `
        <p><b>🌊 Hasil Prediksi:</b></p>
        <p style="font-size: 1.2rem; font-weight: bold; color: ${color};">
          ${status}
        </p>
      `;
    } else {
      resultBox.innerHTML = `<p style="color:red;">❌ Gagal memproses prediksi</p>`;
      console.error("Error detail:", result.message);
    }
  } catch (err) {
    console.error("Gagal memproses prediksi:", err);
    document.getElementById("result").innerHTML = `<p style="color:red;">⚠️ Terjadi kesalahan koneksi.</p>`;
  }
});

