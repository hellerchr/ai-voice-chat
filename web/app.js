const $ = id => document.getElementById(id);
const VOICE_RATE = 24000;  // the server's reply audio: 16-bit mono PCM
let socket, connected = false, ready = false, listening = false;
let stream, inputContext, outputContext, mic;
let nextAt = 0, playing = new Set(), turnEndedAt = 0, measured = true, reply = null;

function state(label, kind = '') {
  $('status').textContent = label;
  $('status-dot').className = `dot ${kind}`;
  $('talk').disabled = !ready || !connected;
  $('talk').setAttribute('aria-pressed', String(listening));
  $('talk-label').textContent = !ready ? 'Loading' : listening ? 'Stop' : 'Start';
}
function idle() {
  if (ready && connected) state(listening ? 'Listening' : 'Ready', listening ? 'recording' : '');
}
function error(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}
function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
function message(role, text) {
  $('empty')?.remove();
  const row = document.createElement('div'); row.className = `message ${role}`;
  const speaker = document.createElement('span'); speaker.className = 'speaker';
  speaker.textContent = role === 'user' ? 'YOU' : 'AI';
  const content = document.createElement('p'); content.textContent = text;
  row.append(speaker, content); $('messages').append(row);
  while ($('messages').children.length > 24) $('messages').firstChild.remove();
  return content;
}

async function openAudio() {
  // Both contexts run at the device's own rate: the worklet resamples the microphone to 16 kHz
  // and the browser resamples the 24 kHz reply buffers, which works in Chrome, Safari and Firefox.
  outputContext ??= new AudioContext({latencyHint: 'interactive'});
  await outputContext.resume();
  if (!mic) {
    // Echo cancellation keeps the assistant's own voice out of the microphone.
    stream = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
    inputContext = new AudioContext({latencyHint: 'interactive'});
    await inputContext.audioWorklet.addModule('/static/audio-worklet.js');
    mic = new AudioWorkletNode(inputContext, 'pcm-input');
    mic.port.onmessage = ({data}) => { if (listening && socket?.readyState === WebSocket.OPEN) socket.send(data); };
    inputContext.createMediaStreamSource(stream).connect(mic);
    mic.connect(inputContext.destination);  // the worklet outputs silence
  }
  await inputContext.resume();
}
async function start() {
  error('');
  state('Waiting for the microphone', 'loading');
  try {
    await openAudio();
  } catch (err) {
    error(err.name === 'NotAllowedError' ? 'The microphone is blocked. Allow it in the browser’s site settings.'
      : err.name === 'NotFoundError' ? 'No microphone found.'
      : `Could not open the microphone: ${err.message}`);
    idle(); return;
  }
  stream.getAudioTracks().forEach(track => { track.enabled = true; });
  listening = true;
  mic.port.postMessage('start');
  send({type: 'listen', language: $('language').value});
  idle();
}
function stop() {
  listening = false;
  mic?.port.postMessage('stop');
  stream?.getAudioTracks().forEach(track => { track.enabled = false; });
  stopPlayback(); send({type: 'mute'});
  idle();
}

function play(bytes) {
  const samples = new Int16Array(bytes);
  if (!outputContext || !samples.length) return;
  const buffer = outputContext.createBuffer(1, samples.length, VOICE_RATE);
  const floats = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) floats[i] = samples[i] / 32768;
  const node = outputContext.createBufferSource();
  node.buffer = buffer; node.connect(outputContext.destination);
  const startAt = Math.max(nextAt, outputContext.currentTime + .03);
  nextAt = startAt + buffer.duration;
  if (!playing.size) { send({type: 'playback', active: true}); state('Speaking', 'recording'); }
  playing.add(node);
  node.onended = () => {
    playing.delete(node);
    if (!playing.size) { send({type: 'playback', active: false}); idle(); }
  };
  node.start(startAt);
  if (!measured) {
    measured = true;
    const delay = (startAt - outputContext.currentTime + (outputContext.outputLatency || outputContext.baseLatency || 0)) * 1000;
    $('latency').textContent = `Answered after ${((performance.now() + delay - turnEndedAt) / 1000).toFixed(2)} s`;
  }
}
function stopPlayback() {
  for (const node of playing) { node.onended = null; try { node.stop(); } catch {} }
  playing.clear(); nextAt = 0;
  send({type: 'playback', active: false});
}

function receive(event) {
  if (event.data instanceof ArrayBuffer) return play(event.data);
  const data = JSON.parse(event.data);
  if (data.type === 'speech') {  // the user talks: the assistant stops at once
    stopPlayback(); reply?.parentNode.classList.remove('pending'); reply = null;
    state('Listening', 'recording');
  } else if (data.type === 'thinking') {
    turnEndedAt = performance.now(); measured = false;
    state('Thinking', 'loading');
  } else if (data.type === 'transcript') {
    message('user', data.text);
    reply = message('assistant', ''); reply.parentNode.classList.add('pending');
  } else if (data.type === 'reply') {
    if (reply) reply.textContent += data.text;
  } else if (data.type === 'done') {
    reply?.parentNode.classList.remove('pending'); reply = null;
    if (!playing.size) idle();
  } else if (data.type === 'empty') {
    idle();
  } else if (data.type === 'listening') {
    idle();
  } else if (data.type === 'error') {
    error(data.message); idle();
  }
}
function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => { connected = true; if (listening) send({type: 'listen', language: $('language').value}); idle(); };
  socket.onmessage = receive;
  socket.onclose = () => {
    connected = ready = false; stopPlayback();
    state('Reconnecting', 'offline');
    setTimeout(() => { connect(); checkStatus(); }, 1500);
  };
}
async function checkStatus() {
  try {
    const data = await (await fetch('/api/status')).json();
    ready = data.ready;
    if (data.error) { error(data.error); state('Failed to load', 'offline'); return; }
    if (ready) idle(); else state(data.status, 'loading');
  } catch { ready = false; state('Server not running', 'offline'); }
  if (!ready) setTimeout(checkStatus, 1000);
}

$('talk').addEventListener('click', () => (listening ? stop() : start()));
$('language').addEventListener('change', () => { if (listening) send({type: 'listen', language: $('language').value}); });
document.addEventListener('keydown', event => { if (event.code === 'Escape' && listening) { event.preventDefault(); stop(); } });
window.addEventListener('pagehide', () => { stop(); stream?.getTracks().forEach(track => track.stop()); });
$('reset').addEventListener('click', () => {
  stopPlayback(); send({type: 'reset'}); reply = null;
  $('messages').replaceChildren(); $('latency').textContent = '';
});
connect(); checkStatus();
