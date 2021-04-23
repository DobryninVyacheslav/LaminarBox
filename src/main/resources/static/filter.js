function normPressure(value) {
    return (value - 400) / 145.09525;
}

async function run() {

    const model = await tf.loadLayersModel('/model/filter/model.json');

    let glass;
    if (document.getElementById("glass").checked) {
        glass = 0.974679;
    } else {
        glass = -0.974679;
    }
    let air;
    if (document.getElementById("air").checked) {
        air = 0.974679;
    } else {
        air = -0.974679;
    }

    const pressure = document.getElementById('pressure').value;

    let result = model.predict(tf.tensor([[glass, air, normPressure(pressure)]])).dataSync();

    document.getElementById('result').innerText =
        Number(result).toFixed(2) + " мин.";

}

document.getElementById("get-time").addEventListener("click", () => {
    run();
})
