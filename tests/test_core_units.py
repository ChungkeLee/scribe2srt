#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

from api.client import Uploader
from core.async_chunk_processor import AsyncChunkProcessor, ChunkProcessorTask, dedupe_boundary_words
from core.intelligent_merger import IntelligentMerger
from core.language_utils import is_cjk_language, normalize_language_code
from core.punctuation_handler import PunctuationHandler
from core.sentence_splitter import SentenceSplitter
from core.srt_processor import SrtProcessor, create_srt_from_json
from core.worker import Worker


def test_uploader_run_retries_before_success(monkeypatch):
    uploader = Uploader(
        file_path="placeholder.mp3",
        payload={"file": ("placeholder.mp3", None, "audio/mp3")},
        headers={},
        max_retries=3,
    )
    calls = []
    finished = []
    errors = []

    def fake_execute():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return {"ok": True}

    monkeypatch.setattr(uploader, "execute", fake_execute)
    monkeypatch.setattr(uploader, "_sleep_before_retry", lambda seconds: None)
    uploader.signals.finished.connect(lambda data: finished.append(data))
    uploader.signals.error.connect(lambda message: errors.append(message))

    uploader.run()

    assert len(calls) == 2
    assert finished == [{"ok": True}]
    assert errors == []


def test_async_cancel_cancels_active_uploaders():
    processor = AsyncChunkProcessor()

    class DummyUploader:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    uploader = DummyUploader()
    assert processor.register_uploader(uploader)

    processor.cancel()

    assert processor.is_cancelled is True
    assert uploader.cancelled is True
    assert processor.active_uploaders == set()


def test_chunk_processor_executes_uploader_in_current_task():
    processor = AsyncChunkProcessor()
    task = ChunkProcessorTask(
        chunk_index=0,
        chunk_path="placeholder.mp3",
        time_offset=0.0,
        language_code="en",
        tag_audio_events=False,
        ffmpeg_available=False,
        max_retries=1,
        parent_processor=processor,
    )

    class DummyUploader:
        def __init__(self):
            self.executed = False
            self.cancelled = False

        def execute(self):
            self.executed = True
            return {"text": "ok", "words": []}

        def cancel(self):
            self.cancelled = True

    uploader = DummyUploader()

    result = task._execute_upload_sync(uploader)

    assert result == {"text": "ok", "words": []}
    assert uploader.executed is True
    assert processor.active_uploaders == set()


def test_chunk_retry_sleep_stops_when_cancelled(monkeypatch):
    processor = AsyncChunkProcessor()
    task = ChunkProcessorTask(
        chunk_index=0,
        chunk_path="placeholder.mp3",
        time_offset=0.0,
        language_code="en",
        tag_audio_events=False,
        ffmpeg_available=False,
        max_retries=2,
        parent_processor=processor,
    )
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        processor.is_cancelled = True

    monkeypatch.setattr("core.async_chunk_processor.time.sleep", fake_sleep)

    try:
        task._sleep_before_retry(5)
    except Exception as exc:
        assert str(exc) == "任务被取消"
    else:
        raise AssertionError("retry sleep should stop when the processor is cancelled")

    assert sleep_calls == [0.2]


def test_async_time_offsets_prefer_recorded_chunk_offsets():
    processor = AsyncChunkProcessor()

    offsets = processor._build_time_offsets(
        chunk_indices=[1, 3, 5],
        split_duration_sec=60,
        chunk_offsets=[0.0, 58.75, 119.0, 179.25],
    )

    assert offsets == {
        1: 58.75,
        3: 179.25,
        5: 300.0,
    }


def test_worker_does_not_fallback_after_async_cancellation(monkeypatch):
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=10,
    )
    worker.async_processor = AsyncChunkProcessor()
    worker.async_processor.completed_chunks[0] = {"text": "done", "words": []}
    worker._is_cancelled = True

    fallback_calls = []
    errors = []
    monkeypatch.setattr(
        worker,
        "_fallback_to_sequential_processing",
        lambda: fallback_calls.append(True),
    )
    worker.error.connect(lambda message: errors.append(message))

    worker._on_async_processing_failed("用户取消了任务")

    assert fallback_calls == []
    assert errors == ["用户取消了任务"]


def test_worker_uses_recorded_offset_for_sequential_chunk(monkeypatch):
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    worker.total_chunks = 3
    worker.current_chunk_index = 1
    worker.temp_chunks = ["chunk-0.mp3", "chunk-1.mp3", "chunk-2.mp3"]
    worker.chunk_offsets = [0.0, 59.321, 121.2]

    processed = []
    monkeypatch.setattr(
        worker,
        "_process_single_file",
        lambda path: processed.append((path, worker.time_offset)),
    )

    worker._process_chunks_sequential()

    assert processed == [("chunk-1.mp3", 59.321)]


def test_worker_offsets_first_processed_restored_chunk(tmp_path, monkeypatch):
    worker = Worker(
        file_path=str(tmp_path / "placeholder.mp3"),
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    worker.total_chunks = 2
    worker.current_chunk_index = 1
    worker.temp_chunks = [
        str(tmp_path / "chunk-0.mp3"),
        str(tmp_path / "chunk-1.mp3"),
    ]
    worker.chunk_offsets = [0.0, 12.5]
    finalized = []
    monkeypatch.setattr(worker, "_finalize_task", lambda: finalized.append(True))

    worker.on_upload_finished({
        "text": "late",
        "words": [
            {"text": "late", "type": "word", "start": 0.2, "end": 0.8}
        ],
    })

    assert finalized == [True]
    assert worker.combined_transcript["words"][0]["start"] == 12.7
    assert worker.combined_transcript["words"][0]["end"] == 13.3


def test_worker_refreshes_combined_audio_duration_metadata():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    worker.combined_transcript = {
        "audio_duration_secs": 10.0,
        "words": [
            {"text": "first", "type": "word", "start": 0.0, "end": 1.0},
            {"text": "last", "type": "word", "start": 42.0, "end": 43.5678},
        ],
    }

    worker._refresh_combined_transcript_metadata()

    assert worker.combined_transcript["audio_duration_secs"] == 43.568


def test_worker_multichunk_json_cleanup_log_matches_cleanup_behavior():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    worker.total_chunks = 2
    logs = []
    worker.log_message.connect(logs.append)

    worker._cleanup_temporary_json_files()

    assert logs == ["多片段处理模式：分片JSON将随临时片段清理"]


def test_worker_smart_split_points_prefer_nearby_silence():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )

    split_points = worker._calculate_smart_split_points(
        duration=180.0,
        silence_ranges=[(57.0, 59.0), (119.5, 121.5)],
    )

    assert split_points == [58.0, 120.5]


def test_worker_splits_slightly_over_threshold_duration():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )

    split_points = worker._calculate_smart_split_points(
        duration=85.0,
        silence_ranges=[],
    )
    ranges = worker._build_segment_ranges(85.0, split_points)

    assert split_points == [60.0]
    assert ranges == [(0.0, 60.0), (60.0, 85.0)]


def test_worker_splits_long_tail_that_would_exceed_threshold():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=40,
    )

    split_points = worker._calculate_smart_split_points(
        duration=2700.0,
        silence_ranges=[],
    )
    ranges = worker._build_segment_ranges(2700.0, split_points)

    assert split_points == [2400.0]
    assert ranges == [(0.0, 2400.0), (2400.0, 2700.0)]


def test_worker_smart_split_does_not_overrun_too_far_for_late_silence():
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )

    split_points = worker._calculate_smart_split_points(
        duration=180.0,
        silence_ranges=[(66.0, 68.0), (126.0, 128.0)],
    )
    ranges = worker._build_segment_ranges(180.0, split_points)

    assert split_points == [60.0, 120.0]
    assert ranges == [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


def test_worker_temp_chunk_dir_requires_owned_chunk_paths(tmp_path):
    source_path = tmp_path / "source.mp3"
    chunk_dir = tmp_path / "source_chunks_test"
    chunk_dir.mkdir()

    worker = Worker(
        file_path=str(source_path),
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )

    owned_chunk = chunk_dir / "source_chunk_000.mp3"
    outside_chunk = tmp_path / "source_chunk_000.mp3"

    assert worker._is_owned_temp_chunk_dir(str(chunk_dir), [str(owned_chunk)])
    assert not worker._is_owned_temp_chunk_dir(str(chunk_dir), [str(outside_chunk)])
    assert not worker._is_owned_temp_chunk_dir(
        str(tmp_path / "source_segments_test"),
        [str(owned_chunk)],
    )


def test_worker_chunk_extension_follows_source_audio_container():
    worker = Worker(
        file_path="placeholder.ogg",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )

    assert worker._chunk_extension_for_audio("source.ogg") == ".ogg"
    assert worker._chunk_extension_for_audio("source.m4a") == ".m4a"
    assert worker._chunk_extension_for_audio("source") == ".mka"


def test_worker_export_audio_segment_uses_stream_copy(monkeypatch):
    worker = Worker(
        file_path="placeholder.ogg",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(worker, "_run_ffmpeg_command", fake_run)

    worker._export_audio_segment("source.ogg", "chunk_000.ogg", 1.25, 4.75)

    command, kwargs = calls[0]
    assert command[command.index("-c:a") + 1] == "copy"
    assert "libmp3lame" not in command
    assert "-b:a" not in command
    assert command[-1] == "chunk_000.ogg"
    assert kwargs["check"] is True


def test_worker_chunk_codec_args_reencode_mp3_and_flac():
    # mp3 帧无法在任意点无缺口流复制、flac 流复制头元数据错误 -> 必须重编码；
    # 其余容器无损流复制。
    worker = Worker(
        file_path="placeholder.m4a",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    assert worker._chunk_codec_args(".mp3") == ["-c:a", "libmp3lame", "-b:a", "192k"]
    assert worker._chunk_codec_args(".flac") == ["-c:a", "flac"]
    assert worker._chunk_codec_args(".MP3") == ["-c:a", "libmp3lame", "-b:a", "192k"]
    for keep_copy in (".ogg", ".m4a", ".aac", ".wav", ".mka"):
        assert worker._chunk_codec_args(keep_copy) == ["-c:a", "copy"]


def test_worker_export_audio_segment_reencodes_mp3(monkeypatch):
    worker = Worker(
        file_path="placeholder.mp3",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    calls = []
    monkeypatch.setattr(worker, "_run_ffmpeg_command",
                        lambda command, **kwargs: calls.append(command))

    worker._export_audio_segment("source.mp3", "chunk_000.mp3", 1.0, 4.0)

    command = calls[0]
    assert command[command.index("-c:a") + 1] == "libmp3lame"
    assert "copy" not in command
    assert "-avoid_negative_ts" in command


def test_language_utils_identifies_cjk_codes():
    assert normalize_language_code("Japanese") == "jap"
    assert is_cjk_language("jpn")
    assert is_cjk_language("zh-CN")
    assert is_cjk_language("kor")
    assert not is_cjk_language("eng")


def test_sentence_splitter_splits_at_sentence_punctuation():
    words = [
        {"text": "Hello ", "type": "word", "start": 0.0, "end": 0.2},
        {"text": "world.", "type": "word", "start": 0.2, "end": 0.5},
        {"text": "Next ", "type": "word", "start": 0.7, "end": 1.0},
        {"text": "line.", "type": "word", "start": 1.0, "end": 1.4},
    ]

    groups = SentenceSplitter("eng").split_into_sentence_groups(words)

    assert len(groups) == 2
    assert "".join(word["text"] for word in groups[0]).strip() == "Hello world."
    assert "".join(word["text"] for word in groups[1]).strip() == "Next line."


def test_intelligent_merger_does_not_cross_complete_sentence():
    merger = IntelligentMerger("eng")
    entry1 = {
        "text": "Hello world.",
        "start": 0.0,
        "end": 1.2,
        "words": [],
        "is_audio_event": False,
    }
    entry2 = {
        "text": "Next line.",
        "start": 1.4,
        "end": 2.4,
        "words": [],
        "is_audio_event": False,
    }

    can_merge, reason = merger._can_merge_entries(entry1, entry2)

    assert can_merge is False
    assert reason == "前一条已是完整句子"


def test_punctuation_handler_sees_sentence_end_before_closer():
    has_punct, punct, priority = PunctuationHandler.word_ends_with_punctuation('"done."')

    assert has_punct is True
    assert punct == "."
    assert priority == 0


def test_srt_processor_adds_terminal_punctuation():
    srt = create_srt_from_json(
        {
            "language_code": "eng",
            "words": [
                {
                    "text": "Hello world",
                    "type": "word",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
        }
    )

    assert "Hello world." in srt


def test_srt_processor_splits_long_single_token_with_bounded_duration():
    captured = {}
    processor = SrtProcessor(
        {
            "language_code": "eng",
            "words": [
                {
                    "text": "abcdefghijklmnop",
                    "type": "word",
                    "start": 0.0,
                    "end": 14.0,
                }
            ],
        }
    )
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    processor.create_srt()

    combined_text = "".join(entry["text"].replace(".", "") for entry in captured["entries"])
    assert combined_text == "abcdefghijklmnop"
    assert len(captured["entries"]) >= 2
    assert all(
        entry["end"] - entry["start"] <= processor.max_subtitle_duration + 0.001
        for entry in captured["entries"]
    )


def test_srt_processor_caps_long_audio_event_duration():
    captured = {}
    processor = SrtProcessor(
        {
            "language_code": "eng",
            "words": [
                {
                    "text": "(noise)",
                    "type": "audio_event",
                    "start": 0.0,
                    "end": 30.0,
                }
            ],
        }
    )
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    processor.create_srt()

    assert captured["entries"][0]["end"] - captured["entries"][0]["start"] <= 7.001


def test_srt_processor_extends_reading_time_backward_before_shifting_next():
    captured = {}
    processor = SrtProcessor(
        {
            "language_code": "eng",
            "words": [
                {"text": "Fast ", "type": "word", "start": 1.0, "end": 1.25},
                {"text": "subtitle ", "type": "word", "start": 1.25, "end": 1.5},
                {"text": "text.", "type": "word", "start": 1.5, "end": 2.0},
                {"text": "Next.", "type": "word", "start": 2.2, "end": 2.6},
            ],
        }
    )
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    processor.create_srt()

    entries = captured["entries"]
    assert entries
    assert all(
        entries[index]["start"] >= entries[index - 1]["end"] - 0.001
        for index in range(1, len(entries))
    )
    assert entries[0]["start"] <= 1.0 + 0.001
    assert entries[-1]["end"] >= 2.6 - 0.001
    assert "Fast subtitle text" in " ".join(entry["text"] for entry in entries)
    assert "Next." in " ".join(entry["text"] for entry in entries)


def test_srt_processor_merges_flash_subtitle_with_neighbor():
    captured = {}
    processor = SrtProcessor(
        {
            "language_code": "eng",
            "words": [
                {"text": "That ", "type": "word", "start": 0.0, "end": 0.25},
                {"text": "helps.", "type": "word", "start": 0.25, "end": 0.8},
                {"text": "uh,", "type": "word", "start": 0.86, "end": 0.94},
                {"text": "thanks.", "type": "word", "start": 1.0, "end": 1.8},
            ],
        }
    )
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    processor.create_srt()

    texts = [entry["text"] for entry in captured["entries"]]
    assert "uh," not in texts
    assert any("uh," in text for text in texts)
    assert all(
        entry["end"] - entry["start"] >= processor.min_subtitle_duration - 0.001
        for entry in captured["entries"]
    )


def test_srt_processor_repairs_tiny_gap_when_room_is_available():
    captured = {}
    processor = SrtProcessor(
        {
            "language_code": "eng",
            "words": [
                {"text": "First ", "type": "word", "start": 0.0, "end": 0.4},
                {"text": "sentence.", "type": "word", "start": 0.4, "end": 1.0},
                {"text": "Second ", "type": "word", "start": 1.03, "end": 1.5},
                {"text": "sentence.", "type": "word", "start": 1.5, "end": 2.8},
            ],
        }
    )
    original_generate = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original_generate(entries)

    processor._generate_final_srt_content = capture
    processor.create_srt()

    entries = captured["entries"]
    for index in range(1, len(entries)):
        gap = entries[index]["start"] - entries[index - 1]["end"]
        assert gap >= processor.min_subtitle_gap - 0.001

    max_late_start = max(
        entry["start"] - processor._entry_time_bounds(entry)[0]
        for entry in entries
    )
    max_late_end = max(
        entry["end"] - processor._entry_time_bounds(entry)[1]
        for entry in entries
    )
    assert max_late_start <= processor._max_late_start_shift() + 0.001
    assert max_late_end <= processor._max_timing_lag() + 0.001


def test_dedupe_boundary_words_drops_overlapping_duplicate():
    # 上一分片尾部 "hello" 结束于 ≈5.02，新分片开头 "hello" 起始于 ≈5.00 —— 同一个跨界词。
    existing = [
        {"text": "say", "type": "word", "start": 4.5, "end": 4.9},
        {"text": "hello", "type": "word", "start": 4.95, "end": 5.02},
    ]
    new = [
        {"text": "hello", "type": "word", "start": 5.00, "end": 5.08},
        {"text": "world", "type": "word", "start": 5.20, "end": 5.60},
    ]

    result = dedupe_boundary_words(existing, new)

    assert [w["text"] for w in result] == ["world"]


def test_dedupe_boundary_words_keeps_legitimate_repeat():
    # 连续重复说出的 "hai"：时间区间不重叠，必须保留两个。
    existing = [
        {"text": "hai", "type": "word", "start": 4.6, "end": 4.9},
    ]
    new = [
        {"text": "hai", "type": "word", "start": 5.10, "end": 5.40},
        {"text": "next", "type": "word", "start": 5.50, "end": 5.90},
    ]

    result = dedupe_boundary_words(existing, new)

    assert [w["text"] for w in result] == ["hai", "next"]


def test_dedupe_boundary_words_noops_without_overlap_or_words():
    assert dedupe_boundary_words([], [{"text": "a", "start": 0, "end": 1}]) == [{"text": "a", "start": 0, "end": 1}]
    existing = [{"text": "a", "type": "word", "start": 0.0, "end": 1.0}]
    assert dedupe_boundary_words(existing, []) == []
    # 文本不同即使时间重叠也不删除
    new = [{"text": "b", "type": "word", "start": 0.5, "end": 1.2}]
    assert dedupe_boundary_words(existing, new) == new


def _cue_times(srt):
    times = []
    for block in [b for b in srt.strip().split("\n\n") if b.strip()]:
        line = block.split("\n")[1]
        a, b = line.split(" --> ")
        def to_s(t):
            h, m, rest = t.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        times.append((to_s(a), to_s(b)))
    return times


def test_subtitle_lead_is_capped_for_fast_speech():
    # 30 个词挤在 1.5s 内：短条需要更长阅读时间，但显示不得提前语音起点太多。
    words = [
        {"type": "word", "text": f"word{i} ",
         "start": round(10.0 + i * 0.05, 3), "end": round(10.0 + i * 0.05 + 0.04, 3)}
        for i in range(30)
    ]
    data = {"language_code": "eng", "words": words}
    processor = SrtProcessor(data)
    srt = processor.create_srt()
    # 语音最早起点 10.0；提前量上限即 processor._max_timing_lead()
    max_lead = processor._max_timing_lead()
    first_start = _cue_times(srt)[0][0]
    assert first_start >= 10.0 - max_lead - 0.01, (first_start, max_lead)


def test_create_srt_is_idempotent_on_same_input():
    # 同一个 dict 连续转换两次必须得到完全相同的结果（预处理不得修改输入对象）。
    data = {
        "language_code": "jpn",
        "words": [
            {"type": "word", "text": "まもなく", "start": 0.0, "end": 1.0},
            {"type": "word", "text": "、", "start": 1.0, "end": 1.05},
            {"type": "word", "text": "到着します", "start": 1.1, "end": 2.4},
        ],
    }
    before = json.dumps(data, ensure_ascii=False, sort_keys=True)
    srt1 = create_srt_from_json(data)
    srt2 = create_srt_from_json(data)
    after = json.dumps(data, ensure_ascii=False, sort_keys=True)
    assert srt1 == srt2
    assert before == after  # 输入对象未被就地修改


def test_refresh_metadata_regenerates_text_from_words():
    # 边界去重后，顶层 text 必须由最终 words 重新生成，二者保持一致。
    worker = Worker(
        file_path="placeholder.m4a",
        language_code="eng",
        tag_audio_events=False,
        max_subtitle_duration=5.0,
        split_duration_min=1,
    )
    worker.combined_transcript = {
        "text": "hello hello world",  # 陈旧、含重复
        "words": [
            {"type": "word", "text": "hello", "start": 0.0, "end": 0.5},
            {"type": "spacing", "text": " ", "start": 0.5, "end": 0.5},
            {"type": "word", "text": "world", "start": 0.6, "end": 1.0},
        ],
    }
    worker._refresh_combined_transcript_metadata()
    assert worker.combined_transcript["text"] == "hello world"
    assert worker.combined_transcript["audio_duration_secs"] == 1.0
