#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.ffmpeg_utils import get_media_info
from core.worker import Worker


def _worker(path="placeholder.mka", split_duration_min=1):
    return Worker(
        file_path=path,
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=split_duration_min,
    )


def test_get_media_info_returns_container_and_audio_details(monkeypatch):
    probe_output = {
        "streams": [{
            "codec_name": "AAC",
            "duration": "12.250",
            "bit_rate": "127999",
            "sample_rate": "48000",
            "channels": 2,
        }],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.300",
            "bit_rate": "130000",
        },
    }
    captured = {}

    monkeypatch.setattr("core.ffmpeg_utils.is_ffmpeg_available", lambda: True)

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(stdout=json.dumps(probe_output))

    monkeypatch.setattr("core.ffmpeg_utils.subprocess.run", fake_run)

    info = get_media_info("voice.m4a")

    assert info == {
        "duration": 12.3,
        "codec": "aac",
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "bit_rate": 127999,
        "sample_rate": 48000,
        "channels": 2,
    }
    show_entries = captured["command"][captured["command"].index("-show_entries") + 1]
    assert "codec_name" in show_entries
    assert "format_name" in show_entries
    assert "bit_rate" in show_entries


@pytest.mark.parametrize(
    ("source_path", "media_info", "extension", "encoder", "stream_copy"),
    [
        (
            "misleading.mka",
            {"codec": "mp3", "format": "matroska,webm", "bit_rate": 64000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.any",
            {"codec": "flac", "format": "flac", "bit_rate": 700000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.mp4",
            {"codec": "aac", "format": "mov,mp4,m4a,3gp,3g2,mj2", "bit_rate": 128000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.aac",
            {"codec": "aac", "format": "aac", "bit_rate": 128000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.ogg",
            {"codec": "opus", "format": "ogg", "bit_rate": 96000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.ogg",
            {"codec": "vorbis", "format": "ogg", "bit_rate": 128000},
            ".flac",
            "flac",
            False,
        ),
        (
            "source.wav",
            {"codec": "pcm_s24le", "format": "wav", "bit_rate": 2304000},
            ".wav",
            "pcm_s24le",
            False,
        ),
    ],
)
def test_chunk_profile_uses_actual_codec_and_container(
        source_path, media_info, extension, encoder, stream_copy):
    profile = _worker(source_path)._select_chunk_export_profile(source_path, media_info)

    assert profile.extension == extension
    assert profile.codec_args[profile.codec_args.index("-c:a") + 1] == encoder
    assert profile.stream_copy is stream_copy


@pytest.mark.parametrize("source_bitrate", [64000, 128000, 320000, None])
def test_mp3_uses_lossless_chunk_output_without_additional_loss(source_bitrate):
    profile = _worker("source.mp3")._select_chunk_export_profile(
        "source.mp3",
        {"codec": "mp3", "format": "mp3", "bit_rate": source_bitrate},
    )

    assert profile.extension == ".flac"
    assert profile.codec_args == ("-c:a", "flac")
    assert profile.stream_copy is False


def test_export_reencode_uses_sample_accurate_trim_and_resets_timestamps(monkeypatch):
    worker = _worker("source.mka")
    profile = worker._select_chunk_export_profile(
        "source.mka",
        {"codec": "mp3", "format": "matroska,webm", "bit_rate": 64000},
    )
    calls = []
    monkeypatch.setattr(worker, "_run_ffmpeg_command", lambda command, **kwargs: calls.append(command))

    worker._export_audio_segment(
        "source.mka", "chunk_000.mp3", 60.0, 120.0, profile=profile
    )

    command = calls[0]
    assert command[command.index("-ss") + 1] == "59.000"
    assert command[command.index("-c:a") + 1] == "flac"
    audio_filter = command[command.index("-af") + 1]
    assert "atrim=start=1.000:end=61.000" in audio_filter
    assert "asetpts=PTS-STARTPTS" in audio_filter


def test_export_m4a_aac_uses_lossless_precise_trim(monkeypatch):
    worker = _worker("source.m4a")
    profile = worker._select_chunk_export_profile(
        "source.m4a",
        {
            "codec": "aac",
            "format": "mov,mp4,m4a,3gp,3g2,mj2",
            "bit_rate": 128000,
        },
    )
    calls = []
    monkeypatch.setattr(worker, "_run_ffmpeg_command", lambda command, **kwargs: calls.append(command))

    worker._export_audio_segment(
        "source.m4a", "chunk_000.flac", 60.0, 120.0, profile=profile
    )

    command = calls[0]
    assert "-ss" not in command
    assert command[command.index("-c:a") + 1] == "flac"
    assert "atrim=start=60.000:end=120.000" in command[command.index("-af") + 1]


def test_split_audio_uses_profile_extension_not_source_extension(monkeypatch, tmp_path):
    source_path = tmp_path / "disguised.mka"
    source_path.write_bytes(b"placeholder")
    worker = _worker(str(source_path))
    captured = []

    monkeypatch.setattr(
        "core.worker.get_media_info",
        lambda path, callback=None, cancel_check=None: {
            "duration": 85.0,
            "codec": "mp3",
            "format": "matroska,webm",
            "bit_rate": 64000,
        },
    )
    monkeypatch.setattr(worker, "_detect_silence_ranges", lambda path, duration: [])

    def fake_export(audio_path, export_jobs, profile):
        for chunk_path, _, _ in export_jobs:
            captured.append((chunk_path, profile))
            Path(chunk_path).write_bytes(b"chunk")

    monkeypatch.setattr(worker, "_export_audio_segments", fake_export)

    assert worker._split_audio(str(source_path)) is True
    assert len(captured) == 2
    assert all(path.endswith(".flac") for path, _ in captured)
    assert all(profile.codec_args[-1] == "flac" for _, profile in captured)
    assert worker.chunk_offsets == [0.0, 60.0]


def test_extremely_short_tail_is_rebalanced_without_changing_normal_tail():
    worker = _worker(split_duration_min=1)

    short_tail_points = worker._calculate_smart_split_points(62.501, [])
    short_tail_ranges = worker._build_segment_ranges(62.501, short_tail_points)
    normal_tail_points = worker._calculate_smart_split_points(85.0, [])

    assert short_tail_points != [60.0]
    assert len(short_tail_ranges) == 2
    assert min(end - start for start, end in short_tail_ranges) >= 30.0
    assert normal_tail_points == [60.0]


def test_long_threshold_short_tail_is_balanced_into_practical_segments():
    worker = _worker(split_duration_min=40)

    points = worker._calculate_smart_split_points(2405.001, [])
    ranges = worker._build_segment_ranges(2405.001, points)

    assert len(ranges) == 2
    assert min(end - start for start, end in ranges) > 1200.0
    assert max(end - start for start, end in ranges) < 1210.0


def test_duration_just_above_threshold_still_produces_multiple_segments():
    worker = _worker(split_duration_min=40)

    points = worker._calculate_smart_split_points(2401.0, [])
    ranges = worker._build_segment_ranges(2401.0, points)

    assert len(ranges) == 2
    assert max(end - start for start, end in ranges) <= 1201.0


def test_unsupported_pcm_container_combination_falls_back_to_flac():
    profile = _worker("source.caf")._select_chunk_export_profile(
        "source.caf",
        {"codec": "pcm_s16be", "format": "caf"},
    )

    assert profile.extension == ".flac"
    assert profile.codec_args == ("-c:a", "flac")


def test_active_ffmpeg_process_is_terminated_on_cancellation(monkeypatch):
    worker = _worker()
    started = threading.Event()

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.args = args[0]
            self.returncode = None
            self.terminated = False
            started.set()

        def communicate(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = 255
            return "", "terminated"

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr("core.worker.subprocess.Popen", FakeProcess)
    errors = []

    def run_command():
        try:
            worker._run_ffmpeg_command(["ffmpeg", "-version"])
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=run_command)
    thread.start()
    assert started.wait(timeout=1.0)
    worker.request_cancellation()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == ["任务已取消。"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration test requires ffmpeg and ffprobe",
)
@pytest.mark.parametrize(
    ("extension", "encoder_args"),
    [
        ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
        ("flac", ["-c:a", "flac"]),
        ("ogg", ["-c:a", "libopus", "-b:a", "64k"]),
        ("m4a", ["-c:a", "aac", "-b:a", "96k"]),
        ("aac", ["-c:a", "aac", "-b:a", "96k"]),
    ],
)
def test_real_exports_do_not_drop_decoded_audio(
        tmp_path, extension, encoder_args):
    sample_rate = 48000
    source_path = tmp_path / f"source.{extension}"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "anoisesrc=color=white:sample_rate=48000:duration=4:seed=1234",
            "-ac", "1", *encoder_args, "-y", str(source_path),
        ],
        check=True,
    )

    worker = _worker(str(source_path))
    media_info = get_media_info(str(source_path))
    profile = worker._select_chunk_export_profile(str(source_path), media_info)
    measured_duration = media_info["duration"]
    midpoint = measured_duration / 2.0
    chunk_paths = []
    export_jobs = []
    for index, (start, end) in enumerate(((0.0, midpoint), (midpoint, measured_duration))):
        chunk_path = tmp_path / f"chunk_{index:03d}{profile.extension}"
        chunk_paths.append(chunk_path)
        export_jobs.append((str(chunk_path), start, end))
    worker._export_audio_segments(str(source_path), export_jobs, profile)

    def decoded_samples(path):
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path), "-f", "s16le", "-ac", "1",
                "-ar", str(sample_rate), "-",
            ],
            capture_output=True,
            check=True,
        )
        return len(result.stdout) // 2

    source_samples = decoded_samples(source_path)
    chunk_samples = sum(decoded_samples(path) for path in chunk_paths)
    # Container decoders may expose encoder tail padding beyond the nominal media
    # duration (notably M4A/AAC).  The chunks must cover at least all nominal
    # samples, without requiring that non-content padding be preserved.
    nominal_samples = round(measured_duration * sample_rate)
    assert chunk_samples >= min(source_samples, nominal_samples)

    if extension == "flac":
        for chunk_path in chunk_paths:
            info = get_media_info(str(chunk_path))
            assert abs(info["duration"] - midpoint) <= 0.001
