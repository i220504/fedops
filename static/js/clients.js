// ==========================================================
//  CLIENT DATA DISTRIBUTION — PRO 3D PIE CHART
// ==========================================================

document.addEventListener("DOMContentLoaded", () => {
    loadDistribution();
});

async function loadDistribution() {
    const loading = document.getElementById("dist-loading");
    const empty = document.getElementById("dist-empty");
    const card = document.getElementById("dist-card");
    const kpiBox = document.getElementById("kpi-summary");

    loading.classList.remove("d-none");

    // ---- Fetch Data ----
    let data = [];
    try {
        const res = await fetch("/api/distribution");
        data = await res.json();
    } catch (e) {
        console.error(e);
    }

    loading.classList.add("d-none");

    // ---- Empty State ----
    if (!data || data.length === 0) {
        empty.classList.remove("d-none");
        return;
    }

    // ---- Extract ----
    const clients = data.map(x => `Client ${x.client}`);
    const values  = data.map(x => x["num-examples"]);

    // ==========================================================
    //  KPI SUMMARY (NOW WORKING)
    // ==========================================================
    kpiBox.classList.remove("d-none");

    document.getElementById("kpi-total-clients").innerText = clients.length;
    document.getElementById("kpi-total-samples").innerText = values.reduce((a, b) => a + b, 0);

    let maxSamples = Math.max(...values);
    let maxIndex   = values.indexOf(maxSamples);
    document.getElementById("kpi-largest-client").innerText =
        `Client ${data[maxIndex].client}`;

    // ---- Show Chart ----
    card.classList.remove("d-none");

    const ctx = document.getElementById("dataDistChart").getContext("2d");

    // 3D shadow style
    const sliceShadow = {
        shadowColor: "rgba(0,0,0,0.35)",
        shadowBlur: 18,
        shadowOffsetX: 5,
        shadowOffsetY: 8
    };

    // Generate gradients
    const gradients = clients.map((_, i) => {
        const g = ctx.createLinearGradient(0, 0, 160, 160);
        g.addColorStop(0, randomBright(i));
        g.addColorStop(1, randomDark(i));
        return g;
    });

    // ==========================================================
    //  CREATE PRO 3D PIE CHART
    // ==========================================================
    new Chart(ctx, {
        type: "pie",
        plugins: [{
            id: "3d-effect",
            beforeDraw(chart) {
                const { ctx } = chart;
                ctx.save();
                ctx.shadowColor = sliceShadow.shadowColor;
                ctx.shadowBlur = sliceShadow.shadowBlur;
                ctx.shadowOffsetX = sliceShadow.shadowOffsetX;
                ctx.shadowOffsetY = sliceShadow.shadowOffsetY;
            },
            afterDraw(chart) {
                chart.ctx.restore();
            }
        }],
        data: {
            labels: clients,
            datasets: [{
                data: values,
                backgroundColor: gradients,
                borderColor: "#fff",
                borderWidth: 2,
                hoverOffset: 18
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: 20 },
            plugins: {
                legend: {
                    position: "right",
                    labels: {
                        padding: 15,
                        font: { size: 14, family: "Inter" },
                        color: "#333"
                    }
                },
                tooltip: {
                    backgroundColor: "#1e1e1e",
                    bodyColor: "#ddd",
                    titleColor: "#fff",
                    padding: 12,
                    borderColor: "#444",
                    borderWidth: 1,
                    callbacks: {
                        label: item => `${item.raw} samples`
                    }
                }
            },
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 1500,
                easing: "easeOutQuart"
            }
        }
    });
}


// ----------------------------------------------------------
//  BEAUTIFUL MODERN GRADIENT COLORS
// ----------------------------------------------------------
function randomBright(i) {
    const colors = [
        "#6a11cb", "#ff512f", "#1d976c", "#f7971e",
        "#4b79a1", "#8360c3", "#de6161", "#11998e",
        "#fc5c7d", "#43cea2"
    ];
    return colors[i % colors.length];
}

function randomDark(i) {
    const colors = [
        "#2575fc", "#dd2476", "#93f9b9", "#ffd200",
        "#283e51", "#2ebf91", "#2657eb", "#38ef7d",
        "#6a82fb", "#185a9d"
    ];
    return colors[i % colors.length];
}
