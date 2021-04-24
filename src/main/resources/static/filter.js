function normPressure(value) {
    const minPressure = 200;
    const maxPressure = 600;
    return (value - minPressure) / (maxPressure - minPressure);
}

async function run() {

    const model = await tf.loadLayersModel('/model/filter/model.json');

    const glass = document.getElementById("glass").checked ? 1.0 : 0.0;
    const air = document.getElementById("air").checked ? 1.0 : 0.0;
    const pressure = normPressure(document.getElementById('pressure').value);

    const result = model.predict(tf.tensor([[glass, air, pressure]])).dataSync();

    document.getElementById('result').innerText =
        Number(result).toFixed(2) + " мин.";

}

document.getElementById("get-time").addEventListener("click", () => {
    run();
})
