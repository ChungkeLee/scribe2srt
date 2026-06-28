#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from api.client import Uploader
from core.async_chunk_processor import AsyncChunkProcessor, ChunkProcessorTask
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

    first, second = captured["entries"][:2]
    assert first["start"] < 1.0
    assert first["end"] <= 2.0 + 0.001
    assert second["start"] <= 2.2 + 0.001
