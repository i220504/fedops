console.log("Dashboard loaded...");

async function loadTrain() {
    const res = await fetch("/api/train");
    return res.json();
}

async function loadEval() {
    const res = await fetch("/api/eval");
    return res.json();
}

async function loadEvents() {
    const res = await fetch("/api/events");
    return res.json();
}

Promise.all([loadTrain(), loadEval(), loadEvents()]).then(([trainData, evalData, eventData]) => {

    // Extract data
    const rounds = trainData.map(x => x.round);
    const trainLoss = trainData.map(x => x.train_loss);
    const evalAcc = evalData.map(x => x.eval_acc);
    const clientCount = eventData.map(x => x.client || 0);

    // ------------------------------
    // Loss Chart
    // ------------------------------
    new Chart(document.getElementById("lossChart"), {
        type: "line",
        data: {
            labels: rounds,
            datasets: [{
                label: "Training Loss",
                data: trainLoss,
                borderColor: "#ff4d4d",
                tension: 0.3,
                borderWidth: 3
            }]
        }
    });

    // ------------------------------
    // Accuracy Chart
    // ------------------------------
    new Chart(document.getElementById("accChart"), {
        type: "line",
        data: {
            labels: rounds,
            datasets: [{
                label: "Eval Accuracy",
                data: evalAcc,
                borderColor: "#007bff",
                borderWidth: 3,
                tension: 0.3
            }]
        }
    });

    // ------------------------------
    // Client Participation
    // ------------------------------
    new Chart(document.getElementById("clientChart"), {
        type: "bar",
        data: {
            labels: rounds,
            datasets: [{
                label: "Clients per Round",
                data: clientCount,
                backgroundColor: "#28a745"
            }]
        }
    });

});
