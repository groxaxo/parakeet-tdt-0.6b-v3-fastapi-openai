(() => {
  const connectBtn = document.getElementById("connectBtn");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const commitBtn = document.getElementById("commitBtn");
  const modelSelect = document.getElementById("modelSelect");
  const languageSelect = document.getElementById("languageSelect");
  const contextInput = document.getElementById("contextInput");
  const silenceBreakInput = document.getElementById("silenceBreakInput");
  const silenceBreakValue = document.getElementById("silenceBreakValue");
  const newLinePerFinalInput = document.getElementById("newLinePerFinalInput");
  const vadThresholdInput = document.getElementById("vadThresholdInput");
  const vadThresholdValue = document.getElementById("vadThresholdValue");
  const partialCooldownInput = document.getElementById("partialCooldownInput");
  const partialCooldownValue = document.getElementById("partialCooldownValue");
  const minPartialInput = document.getElementById("minPartialInput");
  const minPartialValue = document.getElementById("minPartialValue");
  const maxPartialWindowInput = document.getElementById("maxPartialWindowInput");
  const maxPartialWindowValue = document.getElementById("maxPartialWindowValue");
  const minFinalInput = document.getElementById("minFinalInput");
  const minFinalValue = document.getElementById("minFinalValue");
  const echoCancellationInput = document.getElementById("echoCancellationInput");
  const noiseSuppressionInput = document.getElementById("noiseSuppressionInput");
  const autoGainControlInput = document.getElementById("autoGainControlInput");
  const statusEl = document.getElementById("status");
  const wsBadge = document.getElementById("wsBadge");
  const transcriptBox = document.getElementById("transcriptBox");

  let ws = null;
  let audioCtx = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let pendingPartial = "";
  let finalUtterances = [];
  let currentUttId = null;
  let lastResponseCreateMs = 0;
  let newLinePerFinal = false;

  function setStatus(text, ok = false) {
    statusEl.firstChild.textContent = text + " ";
    wsBadge.className = ok ? "badge ok" : "badge warn";
  }

  function renderTranscript() {
    const separator = newLinePerFinal ? "\n" : " ";
    const finalText = finalUtterances.join(separator).trim();
    const merged = [finalText, pendingPartial].filter(Boolean).join(separator).trim();
    const withTrailingNewline =
      newLinePerFinal && finalText.length > 0 && pendingPartial.length === 0 ? `${merged}\n` : merged;
    transcriptBox.value = withTrailingNewline;
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
  }

  function splitFinalSentences(text) {
    const matches = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
    if (!matches) return [];
    return matches.map((s) => s.trim()).filter(Boolean);
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/v1/realtime`;
  }

  function normalizedLanguage() {
    const value = String(languageSelect.value || "autodetect").trim().toLowerCase();
    if (value === "en" || value === "nl") return value;
    return "autodetect";
  }

  function normalizedSilenceBreakMs() {
    const raw = Number(silenceBreakInput.value);
    const clamped = Math.max(300, Math.min(2000, Number.isFinite(raw) ? raw : 900));
    return Math.round(clamped / 50) * 50;
  }

  function clampRound(raw, min, max, step) {
    const numeric = Number(raw);
    const clamped = Math.max(min, Math.min(max, Number.isFinite(numeric) ? numeric : min));
    return Math.round(clamped / step) * step;
  }

  function normalizedVadThreshold() {
    return Number(clampRound(vadThresholdInput.value, 0.002, 0.06, 0.001).toFixed(3));
  }

  function normalizedPartialCooldown() {
    return Number(clampRound(partialCooldownInput.value, 0.2, 2.0, 0.1).toFixed(1));
  }

  function normalizedMinPartialSeconds() {
    return Number(clampRound(minPartialInput.value, 0.2, 3.0, 0.1).toFixed(1));
  }

  function normalizedMaxPartialWindowSeconds() {
    return Number(clampRound(maxPartialWindowInput.value, 1.0, 12.0, 0.5).toFixed(1));
  }

  function normalizedMinFinalSeconds() {
    return Number(clampRound(minFinalInput.value, 0.1, 2.0, 0.1).toFixed(1));
  }

  function updateSilenceBreakLabel() {
    silenceBreakValue.textContent = `${normalizedSilenceBreakMs()} ms`;
  }

  function updateNumericLabels() {
    updateSilenceBreakLabel();
    vadThresholdValue.textContent = normalizedVadThreshold().toFixed(3);
    partialCooldownValue.textContent = `${normalizedPartialCooldown().toFixed(1)} s`;
    minPartialValue.textContent = `${normalizedMinPartialSeconds().toFixed(1)} s`;
    maxPartialWindowValue.textContent = `${normalizedMaxPartialWindowSeconds().toFixed(1)} s`;
    minFinalValue.textContent = `${normalizedMinFinalSeconds().toFixed(1)} s`;
  }

  function sendEvent(event) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(event));
  }

  function sendSessionUpdate() {
    const silenceBreakMs = normalizedSilenceBreakMs();
    silenceBreakInput.value = String(silenceBreakMs);
    const vadThreshold = normalizedVadThreshold();
    vadThresholdInput.value = vadThreshold.toFixed(3);
    const partialCooldown = normalizedPartialCooldown();
    partialCooldownInput.value = partialCooldown.toFixed(1);
    const minPartialSeconds = normalizedMinPartialSeconds();
    minPartialInput.value = minPartialSeconds.toFixed(1);
    const maxPartialWindowSeconds = normalizedMaxPartialWindowSeconds();
    maxPartialWindowInput.value = maxPartialWindowSeconds.toFixed(1);
    const minFinalSeconds = normalizedMinFinalSeconds();
    minFinalInput.value = minFinalSeconds.toFixed(1);
    updateNumericLabels();
    newLinePerFinal = !!newLinePerFinalInput.checked;
    sendEvent({
      type: "session.update",
      session: {
        model: modelSelect.value,
        context_utterances: Math.max(0, Math.min(8, Number(contextInput.value) || 0)),
        language: normalizedLanguage(),
        silence_break_ms: silenceBreakMs,
        new_line_per_final: newLinePerFinal,
        vad_silence_rms_threshold: vadThreshold,
        partial_cooldown_seconds: partialCooldown,
        min_partial_seconds: minPartialSeconds,
        max_partial_window_seconds: maxPartialWindowSeconds,
        min_final_seconds: minFinalSeconds,
      },
    });
  }

  connectBtn.onclick = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
      return;
    }

    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setStatus("Connected", true);
      connectBtn.textContent = "Disconnect";
      startBtn.disabled = false;
      commitBtn.disabled = false;
      sendSessionUpdate();
    };

    ws.onclose = () => {
      setStatus("Disconnected", false);
      connectBtn.textContent = "Connect";
      startBtn.disabled = true;
      stopBtn.disabled = true;
      commitBtn.disabled = true;
      stopMic();
    };

    ws.onerror = () => {
      setStatus("WebSocket error", false);
    };

    ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (_) {
        return;
      }

      const type = msg.type;
      if (type === "session.created" || type === "session.updated") {
        const session = msg.session || {};
        if (session.model) {
          modelSelect.value = session.model;
        }
        if (typeof session.context_utterances === "number") {
          contextInput.value = String(Math.max(0, Math.min(8, Math.trunc(session.context_utterances))));
        }
        if (typeof session.language === "string") {
          const serverLanguage = session.language.toLowerCase();
          languageSelect.value = serverLanguage === "en" || serverLanguage === "nl" ? serverLanguage : "autodetect";
        }
        if (typeof session.silence_break_ms === "number") {
          const ms = Math.max(300, Math.min(2000, Math.round(session.silence_break_ms / 50) * 50));
          silenceBreakInput.value = String(ms);
        }
        if (typeof session.vad_silence_rms_threshold === "number") {
          const value = Math.max(0.002, Math.min(0.06, session.vad_silence_rms_threshold));
          vadThresholdInput.value = value.toFixed(3);
        }
        if (typeof session.partial_cooldown_seconds === "number") {
          const value = Math.max(0.2, Math.min(2.0, session.partial_cooldown_seconds));
          partialCooldownInput.value = value.toFixed(1);
        }
        if (typeof session.min_partial_seconds === "number") {
          const value = Math.max(0.2, Math.min(3.0, session.min_partial_seconds));
          minPartialInput.value = value.toFixed(1);
        }
        if (typeof session.max_partial_window_seconds === "number") {
          const value = Math.max(1.0, Math.min(12.0, session.max_partial_window_seconds));
          maxPartialWindowInput.value = value.toFixed(1);
        }
        if (typeof session.min_final_seconds === "number") {
          const value = Math.max(0.1, Math.min(2.0, session.min_final_seconds));
          minFinalInput.value = value.toFixed(1);
        }
        if (typeof session.new_line_per_final === "boolean") {
          newLinePerFinalInput.checked = session.new_line_per_final;
          newLinePerFinal = session.new_line_per_final;
          renderTranscript();
        }
        updateNumericLabels();
        setStatus(`Connected (${session.model || modelSelect.value}, ${normalizedLanguage()})`, true);
        return;
      }

      if (type === "response.output_text.delta") {
        currentUttId = msg.utterance_id || currentUttId;
        pendingPartial = msg.text || "";
        renderTranscript();
        return;
      }

      if (type === "response.output_text.done") {
        const text = (msg.text || "").trim();
        if (text.length > 0) {
          const chunks = newLinePerFinal ? splitFinalSentences(text) : [text];
          for (const chunk of chunks) {
            finalUtterances.push(chunk);
          }
        }
        pendingPartial = "";
        currentUttId = null;
        renderTranscript();
        return;
      }

      if (type === "error") {
        const errorMessage = msg.error?.message || "Unknown protocol error";
        setStatus(`Error: ${errorMessage}`, false);
      }
    };
  };

  async function startMic() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setStatus("Connect first", false);
      return;
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: !!echoCancellationInput.checked,
        noiseSuppression: !!noiseSuppressionInput.checked,
        autoGainControl: !!autoGainControlInput.checked,
      },
      video: false,
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    processorNode = audioCtx.createScriptProcessor(4096, 1, 1);

    processorNode.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const floatChunk = e.inputBuffer.getChannelData(0);
      const pcm16 = floatTo16kPcm(floatChunk, audioCtx.sampleRate, 16000);
      const b64 = int16ToBase64(pcm16);

      sendEvent({ type: "input_audio_buffer.append", audio: b64 });

      const now = Date.now();
      if (now - lastResponseCreateMs >= 700) {
        sendEvent({ type: "response.create", response: { modalities: ["text"] } });
        lastResponseCreateMs = now;
      }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    startBtn.disabled = true;
    stopBtn.disabled = false;
    commitBtn.disabled = false;
    setStatus("Recording", true);
  }

  function stopMic() {
    if (processorNode) {
      processorNode.disconnect();
      processorNode.onaudioprocess = null;
      processorNode = null;
    }

    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }

    startBtn.disabled = !ws || ws.readyState !== WebSocket.OPEN;
    stopBtn.disabled = true;
  }

  startBtn.onclick = async () => {
    try {
      await startMic();
    } catch (err) {
      setStatus(`Mic error: ${err.message}`, false);
    }
  };

  stopBtn.onclick = () => {
    stopMic();
    setStatus("Connected", true);
  };

  commitBtn.onclick = () => {
    sendEvent({ type: "input_audio_buffer.commit" });
    sendEvent({ type: "response.create", response: { modalities: ["text"] } });
    setStatus("Committed utterance", true);
  };

  modelSelect.onchange = sendSessionUpdate;
  languageSelect.onchange = sendSessionUpdate;
  contextInput.onchange = sendSessionUpdate;
  silenceBreakInput.oninput = sendSessionUpdate;
  vadThresholdInput.oninput = sendSessionUpdate;
  partialCooldownInput.oninput = sendSessionUpdate;
  minPartialInput.oninput = sendSessionUpdate;
  maxPartialWindowInput.oninput = sendSessionUpdate;
  minFinalInput.oninput = sendSessionUpdate;
  echoCancellationInput.onchange = sendSessionUpdate;
  noiseSuppressionInput.onchange = sendSessionUpdate;
  autoGainControlInput.onchange = sendSessionUpdate;
  newLinePerFinalInput.onchange = () => {
    newLinePerFinal = !!newLinePerFinalInput.checked;
    renderTranscript();
    sendSessionUpdate();
  };

  function floatTo16kPcm(float32, sourceRate, targetRate) {
    if (sourceRate === targetRate) {
      const out = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i += 1) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return out;
    }

    const ratio = sourceRate / targetRate;
    const newLen = Math.max(1, Math.round(float32.length / ratio));
    const out = new Int16Array(newLen);

    for (let i = 0; i < newLen; i += 1) {
      const idx = i * ratio;
      const low = Math.floor(idx);
      const high = Math.min(low + 1, float32.length - 1);
      const frac = idx - low;
      const sample = float32[low] * (1 - frac) + float32[high] * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }

    return out;
  }

  function int16ToBase64(int16Arr) {
    const bytes = new Uint8Array(int16Arr.buffer);
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const sub = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode.apply(null, sub);
    }
    return btoa(binary);
  }

  updateNumericLabels();
  setStatus("Disconnected", false);
})();
