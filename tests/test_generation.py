import wave
from pathlib import Path

import app


def test_unique_output_paths_are_distinct(tmp_path):
    app.OUTPUTS_DIR = tmp_path
    first = app.make_unique_output_path("rock ballad guitar instrumental")
    second = app.make_unique_output_path("rock ballad guitar instrumental")
    assert first != second


def test_wav_generation_creates_file(tmp_path):
    output_path = tmp_path / "demo.wav"
    waveform = app.build_synth_waveform("cinematic rock ballad")
    app.write_wav(output_path, waveform)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_song_uses_high_quality_sample_rate(tmp_path):
    app.OUTPUTS_DIR = tmp_path
    output_path, _ = app.generate_song("rock ballad guitar instrumental")
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 44100
