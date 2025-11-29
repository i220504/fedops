async function loadSystem() {
    const res = await fetch("/api/system");
    return res.json();
}

setInterval(() => {
    loadSystem().then(data => {
        cpuChart.data.datasets[0].data.push(data.cpu);
        ramChart.data.datasets[0].data.push(data.memory);

        cpuChart.update();
        ramChart.update();
    })
}, 2000);

// ------------------------------
// CPU Chart
// ------------------------------
const cpuChart = new Chart(document.getElementById("cpuChart"), {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "CPU Usage (%)",
            data: [],
            borderColor: "#ff7f0e",
            tension: 0.3
        }]
    }
});

// ------------------------------
// RAM Chart
// ------------------------------
const ramChart = new Chart(document.getElementById("ramChart"), {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Memory Usage (%)",
            data: [],
            borderColor: "#2ca02c",
            tension: 0.3
        }]
    }
});
