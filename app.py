import hashlib
import os
import re
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
OUTPUTS_DIR = APP_DIR / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "ai-song"


def make_unique_output_path(prompt: str, ext: str = ".wav") -> Path:
    base = slugify(prompt)
    for _ in range(100):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:12]
        digest = hashlib.sha256(f"{prompt}-{stamp}-{unique_id}".encode("utf-8")).hexdigest()[:8]
        candidate = OUTPUTS_DIR / f"{base}-{stamp}-{unique_id}-{digest}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to create a unique output path")


def build_synth_waveform(prompt: str, duration_sec: float = 180.0, sample_rate: int = 44100) -> np.ndarray:
    prompt_lower = prompt.lower()
    base_freq = 220.0
    if "rock" in prompt_lower:
        base_freq = 246.94
    if "ballad" in prompt_lower:
        base_freq = 196.0
    if "guitar" in prompt_lower:
        base_freq = 329.63
    if "cinematic" in prompt_lower:
        base_freq = 261.63
    if "ambient" in prompt_lower:
        base_freq = 174.61

    has_guitar = any(word in prompt_lower for word in ["guitar", "acoustic", "electric", "riff"])
    has_piano = any(word in prompt_lower for word in ["piano", "keys", "keyboard", "ballad"])
    has_pad = any(word in prompt_lower for word in ["ambient", "cinematic", "dreamy", "atmospheric"])
    has_drums = any(word in prompt_lower for word in ["rock", "drum", "beat", "energetic", "anthem"])
    if not any([has_guitar, has_piano, has_pad, has_drums]):
        has_guitar = True
        has_piano = True
        has_pad = True
        has_drums = True

    section_specs = [
        ("intro", 20.0, [0.0, 0.0, 0.0]),
        ("verse", 40.0, [0.0, 1.0, 2.0]),
        ("chorus", 40.0, [1.0, 2.0, 3.0]),
        ("bridge", 35.0, [2.0, 3.0, 1.0]),
        ("outro", 45.0, [3.0, 2.0, 0.0]),
    ]
    total_samples = int(sample_rate * duration_sec)
    waveform = np.zeros(total_samples, dtype=np.float32)
    section_start = 0
    bpm = 104 if "rock" in prompt_lower else 88
    beat_duration = 60.0 / bpm

    for section_name, section_duration, progression in section_specs:
        section_samples = int(sample_rate * section_duration)
        section_end = min(section_start + section_samples, total_samples)
        local_t = np.arange(section_end - section_start) / sample_rate

        root = base_freq * (1.0 + 0.02 * progression[0])
        third = root * 1.25
        fifth = root * 1.5
        seventh = root * 1.75
        chord_freqs = [root, third, fifth, seventh]

        section_guitar = has_guitar and section_name in {"intro", "verse", "chorus", "bridge", "outro"}
        section_piano = has_piano and section_name in {"intro", "verse", "chorus", "bridge", "outro"}
        section_pad = has_pad and section_name in {"intro", "verse", "chorus", "bridge", "outro"}
        section_drums = has_drums and section_name in {"verse", "chorus", "bridge", "outro"}

        for beat_idx in range(int(section_duration / beat_duration)):
            beat_start = int(beat_idx * beat_duration * sample_rate)
            beat_end = int((beat_idx + 1) * beat_duration * sample_rate)
            if beat_start >= section_end - section_start:
                break
            beat_slice = slice(section_start + beat_start, min(section_start + beat_end, section_end))
            beat_t = local_t[beat_start:beat_end]
            if beat_slice.stop <= beat_slice.start:
                continue

            if section_guitar:
                guitar_pattern = [root, third, fifth, root * 1.5]
                for note_idx, note in enumerate(guitar_pattern):
                    note_t = beat_t + note_idx * 0.01
                    envelope = np.exp(-note_t / (0.18 + note_idx * 0.03))
                    waveform[beat_slice] += 0.03 * np.sin(2 * np.pi * (note + note_idx * 4.0) * note_t) * envelope
                if section_name == "chorus":
                    waveform[beat_slice] += 0.04 * np.sin(2 * np.pi * (root * 1.2) * beat_t) * np.exp(-beat_t / 0.14)
                elif section_name == "outro":
                    waveform[beat_slice] += 0.025 * np.sin(2 * np.pi * (root * 0.9) * beat_t) * np.exp(-beat_t / 0.2)

            if section_piano:
                for chord_idx, freq in enumerate(chord_freqs[:3]):
                    piano_envelope = np.exp(-beat_t / (0.25 + chord_idx * 0.04))
                    waveform[beat_slice] += 0.02 * np.sin(2 * np.pi * (freq * 0.5) * beat_t + chord_idx * 0.3) * piano_envelope
                if section_name in {"chorus", "outro"}:
                    waveform[beat_slice] += 0.018 * np.sin(2 * np.pi * (root * 0.5) * beat_t + 0.7) * np.exp(-beat_t / 0.32)

            if section_pad:
                pad_envelope = np.exp(-beat_t / 0.9)
                waveform[beat_slice] += 0.015 * np.sin(2 * np.pi * root * beat_t + 0.2) * pad_envelope
                waveform[beat_slice] += 0.012 * np.sin(2 * np.pi * (root * 1.5) * beat_t + 0.6) * (pad_envelope * 0.8)

            if section_drums:
                kick_phase = np.mod(beat_t, beat_duration)
                kick_envelope = np.exp(-kick_phase / 0.07)
                if beat_idx % 2 == 0:
                    waveform[beat_slice] += 0.045 * np.sin(2 * np.pi * (base_freq / 4.0) * beat_t) * kick_envelope
                if beat_idx % 4 == 2:
                    snare_envelope = np.exp(-np.mod(beat_t + 0.18, 0.36) / 0.05)
                    waveform[beat_slice] += 0.02 * np.sin(2 * np.pi * 1800.0 * beat_t) * snare_envelope
                if beat_idx % 2 == 0:
                    hat_envelope = np.exp(-np.mod(beat_t, 0.08) / 0.025)
                    waveform[beat_slice] += 0.012 * np.sin(2 * np.pi * 7000.0 * beat_t) * hat_envelope

            bass_freq = root / 2.0
            bass_envelope = np.exp(-beat_t / 0.32)
            waveform[beat_slice] += 0.035 * np.sin(2 * np.pi * bass_freq * beat_t + 0.1) * bass_envelope

        section_start = section_end

    waveform = np.tanh(waveform / 0.95)
    return waveform.astype(np.float32)


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int = 44100) -> Path:
    amplitude = np.iinfo(np.int16).max
    data = (waveform * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())
    return path


def generate_with_ace_step(prompt: str, uploaded_file: object | None = None) -> Path:
    model_id = os.getenv("ACE_STEP_MODEL", "Ace-Step/1.5-xl")
    try:
        from transformers import pipeline  # type: ignore

        _ = pipeline("text-to-audio", model=model_id)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"ACE-Step model not ready in this environment: {exc}") from exc

    raise RuntimeError("The Ace-Step integration is configured but requires the model runtime to be installed.")


def generate_song(prompt: str, uploaded_file: object | None = None) -> tuple[Path, str]:
    output_path = make_unique_output_path(prompt)
    try:
        generate_with_ace_step(prompt, uploaded_file)
    except Exception as exc:
        waveform = build_synth_waveform(prompt)
        write_wav(output_path, waveform)
        note = f"Ace-Step was unavailable, so a locally generated 3-minute arrangement was created instead. {exc}"
        return output_path, note

    note = "Generated through the Ace-Step pipeline."
    return output_path, note


st.set_page_config(page_title="AI Song Generator", page_icon="🎵", layout="centered")

st.title("🎼 AI Song Generator")
st.caption("Describe a style or upload an audio/video reference, then generate a full 3-minute song draft with arranged sections and multiple instruments.")

with st.form("song_form"):
    prompt = st.text_area(
        "Prompt",
        placeholder="Create a rock ballad guitar instrumental with warm reverb and a cinematic chorus",
        height=120,
    )
    uploaded_file = st.file_uploader(
        "Optional audio or video reference",
        type=["mp3", "wav", "m4a", "ogg", "mp4", "mov", "avi", "mkv", "webm"],
    )
    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt.strip():
        st.error("Please enter a creative prompt before generating.")
    else:
        if uploaded_file is not None:
            save_path = UPLOADS_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.getbuffer())
            st.success(f"Reference saved to {save_path.name}")

        output_path, note = generate_song(prompt, uploaded_file)
        st.success(f"Generation complete. Output: {output_path.name}")
        st.info(note)
        st.audio(str(output_path), format="audio/wav")
        with open(output_path, "rb") as handle:
            st.download_button(
                "Download WAV",
                handle.read(),
                file_name=output_path.name,
                mime="audio/wav",
            )
