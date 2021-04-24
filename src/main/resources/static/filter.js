function normPressure(value) {
    return (value - 400) / 130.930734;
}

async function run() {

    const model = await tf.loadLayersModel('/model/filter/model.json');

    let glass;
    let normalizedUnitValue = 0.986013;
    if (document.getElementById("glass").checked) {
        glass = normalizedUnitValue;
    } else {
        glass = -normalizedUnitValue;
    }
    let air;
    if (document.getElementById("air").checked) {
        air = normalizedUnitValue;
    } else {
        air = -normalizedUnitValue;
    }

    const pressure = document.getElementById('pressure').value;

    let result = model.predict(tf.tensor([[glass, air, normPressure(pressure)]])).dataSync();

    document.getElementById('result').innerText =
        Number(result).toFixed(2) + " мин.";

}

document.getElementById("get-time").addEventListener("click", () => {
    run();
})
