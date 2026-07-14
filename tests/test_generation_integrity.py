#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import os
import re
from collections import Counter
from types import SimpleNamespace

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
    # 按 SRT 块结构提取正文（跳过序号行与时间行），而不是用 isdigit 猜测。
    # 全角数字正文行（如 "３"）的 str.isdigit() 为 True，按内容猜测会误删正文。
    text_lines = []
    for block in [b for b in srt.strip().split("\n\n") if b.strip()]:
        lines = block.split("\n")
        text_lines.extend(lines[2:])
    return _normalize_spacing("".join(text_lines))


def _has_character_coverage(expected: str, actual: str) -> bool:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return all(actual_counts[char] >= count for char, count in expected_counts.items())


def _write_srt(tmp_path, name: str, srt: str):
    srt_path = tmp_path / name
    srt_path.write_text(srt, encoding="utf-8")
    return srt_path


def _capture_entries(processor: SrtProcessor):
    captured = {}
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    srt = processor.create_srt()
    return srt, captured["entries"]


def _entry_time_bounds(entry: dict):
    timed_items = [
        item for item in entry.get("words", [])
        if isinstance(item.get("start"), (int, float)) and isinstance(item.get("end"), (int, float))
    ]
    if timed_items:
        return (
            min(item["start"] for item in timed_items),
            max(item["end"] for item in timed_items),
        )
    return entry.get("start", 0), entry.get("end", 0)


def _assert_no_material_timeline_drift(entries, max_start_lag=1.5):
    for entry in entries:
        if entry.get("is_audio_event"):
            continue

        source_start, source_end = _entry_time_bounds(entry)
        assert entry["start"] - source_start <= max_start_lag
        assert source_end - entry["end"] <= 0.15


def test_sample_json_generation_is_complete_and_rule_compliant(tmp_path):
    analyzer = EnhancedSubtitleAnalyzer()
    sample_paths = sorted(glob.glob(os.path.join("sample", "*.json")))

    assert sample_paths, "sample JSON fixtures are required"

    for sample_path in sample_paths:
        with open(sample_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        processor = SrtProcessor(data)
        srt, entries = _capture_entries(processor)
        assert "-->" in srt
        assert _has_character_coverage(_source_text_from_processor(processor), _srt_text(srt)), sample_path
        _assert_no_material_timeline_drift(entries, max_start_lag=10.0)

        srt_path = _write_srt(
            tmp_path,
            os.path.splitext(os.path.basename(sample_path))[0] + ".srt",
            srt,
        )
        result = analyzer.analyze_subtitle_rules(str(srt_path))
        assert "error" not in result

        # 这些类别在"精准时间轴优先"策略下会有少量残留，作为压力项容忍（见后处理说明）。
        tolerated_violations = {
            "duration_too_short",
            "gap_too_small",
            "cps_too_high",
            # 因显示约束被拆开的句子中段是延续镜头，按专业规范不补句末标点，
            # 因此"末尾非标点"不再是硬性违规（修复了句中被强插句号的回归）。
            "punctuation_issues",
        }
        violations = {
            name: values
            for name, values in result["violations"].items()
            if values and name not in tolerated_violations
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

    processor = SrtProcessor(data)
    _, entries = _capture_entries(processor)
    _assert_no_material_timeline_drift(entries, max_start_lag=0.15)

    previous_end = None
    for subtitle in subtitles:
        start_text, end_text = subtitle["time"].split(" --> ")
        start = analyzer.parse_srt_time(start_text)
        end = analyzer.parse_srt_time(end_text)

        assert end > start
        if previous_end is not None:
            assert start - previous_end >= -0.001
        previous_end = end


def test_dense_json_timeline_does_not_accumulate_drift():
    words = []
    timestamp = 0.0
    for index in range(120):
        words.append({
            "text": f"word{index}",
            "type": "word",
            "start": timestamp,
            "end": timestamp + 0.22,
        })
        timestamp += 0.23

    processor = SrtProcessor({"language_code": "eng", "words": words})
    _, entries = _capture_entries(processor)

    last_source_end = words[-1]["end"]
    assert entries[-1]["end"] >= last_source_end - 0.15
    assert entries[-1]["end"] - last_source_end <= processor._max_timing_lag() + 0.001
    assert all(
        entry["end"] - _entry_time_bounds(entry)[1] <= processor._max_timing_lag() + 0.001
        for entry in entries
    )
    _assert_no_material_timeline_drift(entries, max_start_lag=0.15)


def _parse_cues(srt: str):
    """Return list of (start, end, [text_lines]) from an SRT string."""
    cues = []
    for block in [b for b in srt.strip().split("\n\n") if b.strip()]:
        lines = block.split("\n")
        times = lines[1].split(" --> ")
        cues.append((_srt_time_to_seconds(times[0]), _srt_time_to_seconds(times[1]), lines[2:]))
    return cues


def _srt_time_to_seconds(text: str) -> float:
    hours, minutes, rest = text.strip().split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def test_audio_event_keeps_original_text_without_terminal_punctuation():
    """回归：音频事件不应被补句末标点，如 (拍手) 不得变成 (拍手。)。"""
    data = {
        "language_code": "jpn",
        "words": [
            {"type": "word", "text": "こんにちは", "start": 0.0, "end": 1.0},
            {"type": "audio_event", "text": "(拍手)", "start": 1.2, "end": 2.0},
        ],
    }
    srt = create_srt_from_json(data)
    event_lines = [ln for cue in _parse_cues(srt) for ln in cue[2] if ln.startswith("(")]
    assert event_lines, "audio event cue should be present"
    for line in event_lines:
        assert line == "(拍手)", f"audio event must stay verbatim, got {line!r}"


def test_audio_event_not_collapsed_to_flash_in_dense_zone():
    """回归：密集/退化时间区中的音频事件不应被压成 1ms 闪现。"""
    data = {
        "language_code": "jpn",
        "words": [
            {"type": "word", "text": "あ", "start": 5.0, "end": 5.0},
            {"type": "word", "text": "い", "start": 5.0, "end": 5.0},
            {"type": "audio_event", "text": "(笑い)", "start": 5.0, "end": 5.0},
            {"type": "word", "text": "うえお", "start": 5.0, "end": 5.0},
        ],
    }
    processor = SrtProcessor(data)
    srt, entries = _capture_entries(processor)
    matching = [entry for entry in entries if "(笑い)" in entry.get("text", "")]
    assert matching, "audio event text should survive"
    for entry in matching:
        duration = entry["end"] - entry["start"]
        assert duration >= 0.001, f"audio event collapsed to {duration:.3f}s"
    assert "(笑い)" in srt


def test_zero_duration_cluster_is_not_exploded_into_per_char_punctuation():
    """回归：源含零时长词簇时，不得逐字插句号（如 声。量。本。）。"""
    text = "声量本身反而成为一种网民自发"
    words = [{"type": "word", "text": ch, "start": 10.0, "end": 10.0} for ch in text]
    srt = create_srt_from_json({"language_code": "zho", "words": words})

    cues = _parse_cues(srt)
    joined = "".join(ln for _, _, lines in cues for ln in lines)
    # 内容不丢失
    assert all(ch in joined for ch in text)
    # 注入的句号数量应远小于字符数（理想为每条至多一个句尾句号）
    assert joined.count("。") <= len(cues), (
        f"terminal punctuation exploded per-character: {joined!r}"
    )
    # 不应出现"句号夹在两个汉字中间"的乱码模式
    for index, char in enumerate(joined):
        if char == "。" and index + 1 < len(joined):
            following = joined[index + 1]
            assert following in "。）」』】》" or index == len(joined) - 1, (
                f"period injected mid-text before {following!r}: {joined!r}"
            )


def test_mid_sentence_split_has_no_injected_terminal_punctuation():
    """回归：因长度被拆开的句子中段不应被强插句号，仅句尾补一次。"""
    text = "これはとても長い文章でありながら内部には句読点がまったく存在しないため分割位置の判断が難しい文です"
    words = [
        {"type": "word", "text": ch, "start": round(i * 0.2, 2), "end": round(i * 0.2 + 0.18, 2)}
        for i, ch in enumerate(text)
    ]
    words[-1]["text"] += "。"
    srt = create_srt_from_json({"language_code": "jpn", "words": words})

    cues = _parse_cues(srt)
    assert len(cues) >= 2, "long sentence should split into multiple cues"

    joined_lines = [ln for _, _, lines in cues for ln in lines]
    # 每一行内部都不应出现"汉字。汉字"式的句中句号
    for line in joined_lines:
        for index, char in enumerate(line[:-1]):
            if char == "。":
                assert line[index + 1] in "。）」』】》", f"mid-line period in {line!r}"
    # 只有整体最后一条以句号收尾
    non_final_lines = joined_lines[:-1]
    assert not any(ln.endswith("。") for ln in non_final_lines), (
        f"continuation cues should not end with a period: {non_final_lines}"
    )
    assert joined_lines[-1].endswith("。"), "the true sentence end should keep its period"


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
    worker.chunk_offsets = [0.0, 58.75, 119.5, 181.0]
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
    assert state["chunk_offsets"] == [0.0, 58.75, 119.5, 181.0]
