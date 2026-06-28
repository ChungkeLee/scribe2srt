#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import os
import re
from collections import Counter
from types import SimpleNamespace

from core.config import MIN_SUBTITLE_GAP
from core.srt_processor import SrtProcessor, create_srt_from_json
from core.worker import Worker
from tests.optimize_based_on_analysis import EnhancedSubtitleAnalyzer


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _source_text_from_json(data: dict) -> str:
    parts = []
    for word in data.get("words", []):
        if word.get("type") == "spacing":
            continue
        parts.append(word.get("text", ""))
    return _normalize_spacing("".join(parts))


def _source_text_from_processor(processor: SrtProcessor) -> str:
    timeline_items = processor.words + processor.audio_events
    timeline_items.sort(key=lambda item: item.get("start", 0))
    return _normalize_spacing("".join(item.get("text", "") for item in timeline_items))


def _srt_text(srt: str) -> str:
    text_lines = []
    for line in srt.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        text_lines.append(stripped)
    return _normalize_spacing("".join(text_lines))


def _has_character_coverage(expected: str, actual: str) -> bool:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return all(actual_counts[char] >= count for char, count in expected_counts.items())


def _write_srt(tmp_path, name: str, srt: str):
    srt_path = tmp_path / name
    srt_path.write_text(srt, encoding="utf-8")
    return srt_path


def test_sample_json_generation_is_complete_and_rule_compliant(tmp_path):
    analyzer = EnhancedSubtitleAnalyzer()
    sample_paths = sorted(glob.glob(os.path.join("sample", "*.json")))

    assert sample_paths, "sample JSON fixtures are required"

    for sample_path in sample_paths:
        with open(sample_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        processor = SrtProcessor(data)
        srt = processor.create_srt()
        assert "-->" in srt
        assert _has_character_coverage(_source_text_from_processor(processor), _srt_text(srt)), sample_path

        srt_path = _write_srt(
            tmp_path,
            os.path.splitext(os.path.basename(sample_path))[0] + ".srt",
            srt,
        )
        result = analyzer.analyze_subtitle_rules(str(srt_path))
        assert "error" not in result

        violations = {
            name: values
            for name, values in result["violations"].items()
            if values
        }
        assert violations == {}, f"{sample_path}: {violations}"


def test_generated_srt_timeline_is_ordered_and_non_overlapping(tmp_path):
    analyzer = EnhancedSubtitleAnalyzer()
    sample_path = os.path.join("sample", "ElevenLabs.en.json")

    with open(sample_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    srt = create_srt_from_json(data)
    srt_path = _write_srt(tmp_path, "timeline.srt", srt)
    subtitles = analyzer.quality_analyzer.parse_srt_file(str(srt_path))

    previous_end = None
    for subtitle in subtitles:
        start_text, end_text = subtitle["time"].split(" --> ")
        start = analyzer.parse_srt_time(start_text)
        end = analyzer.parse_srt_time(end_text)

        assert end > start
        if previous_end is not None:
            assert start - previous_end >= MIN_SUBTITLE_GAP - 0.001
        previous_end = end


def _transcript(chunk_index: int) -> dict:
    start = chunk_index * 10.0
    return {
        "text": f"chunk-{chunk_index}",
        "words": [
            {
                "text": f"chunk-{chunk_index}",
                "type": "word",
                "start": start,
                "end": start + 1.0,
            }
        ],
    }


def test_fallback_retries_from_first_missing_chunk(monkeypatch):
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="en",
        tag_audio_events=False,
        max_subtitle_duration=7.0,
        split_duration_min=1,
    )
    worker.total_chunks = 4
    worker.temp_chunks = [f"chunk-{index}.mp3" for index in range(worker.total_chunks)]
    worker.async_base_chunk_index = 0
    worker.async_processor = SimpleNamespace(
        completed_chunks={
            0: _transcript(0),
            2: _transcript(2),
        }
    )

    resumed_from = []
    monkeypatch.setattr(worker, "_process_chunks_sequential", lambda: resumed_from.append(worker.current_chunk_index))
    monkeypatch.setattr(worker, "_finalize_task", lambda: resumed_from.append("finalized"))

    worker._fallback_to_sequential_processing()

    assert worker.current_chunk_index == 1
    assert resumed_from == [1]
    assert worker.combined_transcript["text"] == "chunk-0"


def test_restored_fallback_keeps_existing_prefix_and_retries_gap(monkeypatch):
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="en",
        tag_audio_events=False,
        max_subtitle_duration=7.0,
        split_duration_min=1,
    )
    worker.total_chunks = 5
    worker.temp_chunks = [f"chunk-{index}.mp3" for index in range(worker.total_chunks)]
    worker.combined_transcript = _transcript(0)
    worker._append_transcript(_transcript(1))
    worker.async_base_chunk_index = 2
    worker.async_processor = SimpleNamespace(
        completed_chunks={
            2: _transcript(2),
            4: _transcript(4),
        }
    )

    resumed_from = []
    monkeypatch.setattr(worker, "_process_chunks_sequential", lambda: resumed_from.append(worker.current_chunk_index))
    monkeypatch.setattr(worker, "_finalize_task", lambda: resumed_from.append("finalized"))

    worker._fallback_to_sequential_processing()

    assert worker.current_chunk_index == 3
    assert resumed_from == [3]
    assert worker.combined_transcript["text"] == "chunk-0 chunk-1 chunk-2"


def test_async_get_state_uses_restorable_base_index():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="en",
        tag_audio_events=False,
        max_subtitle_duration=7.0,
        split_duration_min=1,
    )
    worker.current_chunk_index = 4
    worker.async_base_chunk_index = 2
    worker.async_processor = SimpleNamespace(
        get_progress_info=lambda: {
            "total_chunks": 3,
            "completed_chunks": 0,
            "failed_chunks": 0,
            "processing_chunks": 1,
            "is_cancelled": False,
        }
    )

    state = worker.get_state()

    assert state["current_chunk_index"] == 2
    assert state["async_progress"]["processing_chunks"] == 1
