let chart; // variabel global agar bisa di-destroy

async function loadDashboardData(tahun = "", provinsi = "", ikan = "") {
  const url = new URL(window.location.origin + "/api/dashboard-populasi");
  if (tahun) url.searchParams.append("tahun", tahun);
  if (provinsi) url.searchParams.append("provinsi", provinsi);
  if (ikan) url.searchParams.append("ikan", ikan);

  const res = await fetch(url);
  const data = await res.json();
  console.log("Data API (filtered):", data);

  // Normalisasi nama ikan (agar cocok meski huruf besar)
  const ikanUtama = ["TUNA", "TONGKOL", "CAKALANG", "MARLIN", "TENGGIRI"];
  const dataFiltered = data.filter(d => ikanUtama.includes(d.kelompok_ikan.toUpperCase()));

  const tahunSet = [...new Set(dataFiltered.map(d => d.tahun))].sort();
  const datasets = ikanUtama.map(ikan => {
    const values = tahunSet.map(th => {
      const found = dataFiltered.find(
        d => d.tahun === th && d.kelompok_ikan.toUpperCase() === ikan
      );
      return found ? found.populasi : null;
    });
    return {
      label: ikan,
      data: values,
      fill: false,
      borderWidth: 2,
      tension: 0.3,
      showLine: true,
      pointRadius: 5
    };
  });

  const ctx = document.getElementById("chartPopulasi").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: tahunSet,
      datasets: datasets
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        title: {
          display: true,
          text: "Tren Populasi 5 Jenis Ikan Utama (per Tahun)"
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Populasi (rata-rata TP_C & TP_E)" }
        },
        x: { title: { display: true, text: "Tahun" } }
      }
    }
  });
}

// === FUNGSI PETA KEPAHUHAN / OVERFISHING ===
async function loadMap(tahun = "", provinsi = "", ikan = "") {
  const url = new URL(window.location.origin + "/api/peta-kepatuhan");
  if (tahun) url.searchParams.append("tahun", tahun);
  if (provinsi) url.searchParams.append("provinsi", provinsi);
  if (ikan) url.searchParams.append("ikan", ikan);

  const res = await fetch(url);
  const data = await res.json();
  document.getElementById("mapContainer").innerHTML = data.html;
}


// === EVENT SAAT HALAMAN SUDAH DIMUAT ===
document.addEventListener("DOMContentLoaded", () => {
  // tampilkan grafik default semua data saat halaman dibuka
  loadDashboardData();
  loadMap();

  // handle klik tombol filter
  document.getElementById("filterForm").addEventListener("submit", e => {
    e.preventDefault();
    const tahun = document.getElementById("tahun").value;
    const provinsi = document.getElementById("provinsi").value;
    const ikan = document.getElementById("ikan").value;
    loadDashboardData(tahun, provinsi, ikan);
    loadMap(tahun, provinsi, ikan); 
  });
});
