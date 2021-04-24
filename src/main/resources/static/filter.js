const minPressure = 200;
const maxPressure = 600;
const onValue = 1.0;
const offValue = 0.0;

function normPressure(value) {
    return (value - minPressure) / (maxPressure - minPressure);
}

async function run() {

    const model = await tf.loadLayersModel('/model/filter/model.json');
    const glass = document.getElementById("glass").checked ? onValue : offValue;
    const air = document.getElementById("air").checked ? onValue : offValue;
    const pressure = document.getElementById('pressure').value;

    const result = pressure < maxPressure ?
        model.predict(tf.tensor([[glass, air, normPressure(pressure)]])).dataSync() : 0.0;

    document.getElementById('result').innerText = Number(result).toFixed(2) + " мин.";

}

document.getElementById("get-time").addEventListener("click", () => {
    run();
})
