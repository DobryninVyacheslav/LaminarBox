const webcamElement = document.getElementById('webcam');
const button = document.getElementById('button');
const consoleElement = document.getElementById('console');
const statusElement = document.getElementById('status');
const select = document.getElementById('select');
let currentStream;

function stopMediaTracks(stream) {
  stream.getTracks().forEach(track => {
    track.stop();
  });
}


function getDevices(mediaDevices) {
  select.innerHTML = '';
  select.appendChild(document.createElement('option'));
  let count = 1;
  mediaDevices.forEach(mediaDevice => {
    if (mediaDevice.kind === 'videoinput') {
      const option = document.createElement('option');
      option.value = mediaDevice.deviceId;
      const label = mediaDevice.label || `Camera ${count++}`;
      const textNode = document.createTextNode(label);
      option.appendChild(textNode);
      select.appendChild(option);
    }
  });
}

navigator.mediaDevices.enumerateDevices().then(getDevices);

button.addEventListener('click', event => {
  if (typeof currentStream !== 'undefined') {
    stopMediaTracks(currentStream);
  }
  const videoConstraints = {
        width: 180,
        height: 180
      };
  if (select.value === '') {
    videoConstraints.facingMode = 'environment';
  } else {
    videoConstraints.deviceId = { exact: select.value };
  }
  const constraints = {
    video: videoConstraints,
    audio: false
  };


  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(stream => {
      currentStream = stream;
      webcamElement.srcObject = stream;
      return navigator.mediaDevices.enumerateDevices();
    })
    .then(getDevices)
    .catch(error => {
      console.error(error);
    });


});


statusElement.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;

const MODEL_URL = 'https://cdn.glitch.com/52a372b6-0a36-4419-aade-90db47e9aed5%2Fmodel.json?v=1618127183721';
const SLEEP_TIME = 500;



function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function run() {
    console.log('Loading model..');
    const model = await tf.loadLayersModel(MODEL_URL);
    console.log('Successfully loaded model');
    console.log(model.summary());

    const classes = ['correct', 'incorrect'];
    const webcam = await tf.data.webcam(webcamElement);

    while (true) {
      const img = await webcam.capture();

      const result = model.predict(img.reshape([1, 180,180,3]));
      const score = result.softmax();
      const axis = 1;
      const argMax = score.argMax(axis);

      consoleElement.innerText = `
        Predictions: ${result}\n
        Score: ${score}\n
        ArgMax: ${argMax}\n
      `;

      statusElement.innerText = `
        Class: ${classes[argMax.dataSync()]}
        Probability: ${100 * score.max().dataSync()}
      `;

      img.dispose();

      await tf.nextFrame();
      await sleep(SLEEP_TIME);
    }
}


run();