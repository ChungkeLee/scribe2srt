from core.srt_processor import SrtProcessor


def _capture_entries(processor):
    captured = {}
    original = processor._generate_final_srt_content

    def capture(entries):
        captured["entries"] = [entry.copy() for entry in entries]
        return original(entries)

    processor._generate_final_srt_content = capture
    srt = processor.create_srt()
    return srt, captured["entries"]


def _source_bounds(entry):
    return (
        min(word["start"] for word in entry["words"]),
        max(word["end"] for word in entry["words"]),
    )


def test_final_word_timings_remain_inside_sync_bounds():
    words = []
    for index in range(80):
        start = index * 0.08
        words.append({
            "type": "word",
            "text": f"word{index} ",
            "start": start,
            "end": start + 0.04,
        })

    processor = SrtProcessor({"language_code": "eng", "words": words})
    _, entries = _capture_entries(processor)

    for entry in entries:
        if entry.get("is_audio_event"):
            continue
        source_start, source_end = _source_bounds(entry)
        assert source_start - entry["start"] <= processor._max_timing_lead() + 0.001
        assert entry["start"] - source_start <= processor._max_late_start_shift() + 0.001
        assert entry["end"] - source_end <= processor._max_timing_lag() + 0.001


def test_zero_duration_cluster_is_not_queued_seconds_late():
    words = [
        {"type": "word", "text": character, "start": 5.0, "end": 5.0}
        for character in "zero duration timestamp cluster"
    ]
    processor = SrtProcessor({"language_code": "eng", "words": words})
    srt, entries = _capture_entries(processor)

    assert entries
    assert max(entry["start"] for entry in entries) <= (
        5.0 + processor._max_late_start_shift() + 0.001
    )
    for block in srt.strip().split("\n\n"):
        start, end = block.splitlines()[1].split(" --> ")
        assert start != end


def test_simultaneous_audio_events_do_not_accumulate_delay():
    data = {
        "language_code": "eng",
        "words": [
            {"type": "audio_event", "text": "[laughs]", "start": 5.0, "end": 5.0},
            {"type": "audio_event", "text": "[claps]", "start": 5.0, "end": 5.0},
            {"type": "audio_event", "text": "[music]", "start": 5.0, "end": 5.0},
        ],
    }
    processor = SrtProcessor(data)
    _, entries = _capture_entries(processor)

    assert len(entries) == 1
    assert entries[0]["start"] == 5.0
    assert entries[0]["end"] - entries[0]["start"] >= processor.min_subtitle_duration - 0.001

    blocks = [block for block in processor.create_srt().strip().split("\n\n") if block]
    assert len(blocks) == 1
    assert "[laughs] [claps] [music]" in blocks[0]


def test_audio_event_overlapping_dialogue_is_co_displayed_not_delayed():
    processor = SrtProcessor({
        "language_code": "eng",
        "words": [
            {"type": "word", "text": "Hello world", "start": 0.0, "end": 1.6},
            {"type": "audio_event", "text": "[laughs]", "start": 0.4, "end": 0.6},
        ],
    })

    srt, entries = _capture_entries(processor)

    assert len(entries) == 1
    assert entries[0]["start"] <= 0.4 <= entries[0]["end"]
    assert "Hello world" in srt
    assert "[laughs]" in srt


def test_long_embedded_event_is_resplit_without_cpl_overflow():
    dialogue = "one two three four five six seven eight nine ten " * 2
    event = "[very long overlapping background laughter event]"
    processor = SrtProcessor({
        "language_code": "eng",
        "words": [
            {"type": "word", "text": dialogue, "start": 0.0, "end": 5.0},
            {"type": "audio_event", "text": event, "start": 2.0, "end": 3.0},
        ],
    })

    srt = processor.create_srt()
    blocks = [block for block in srt.strip().split("\n\n") if block]

    normalized_text = " ".join(
        line
        for block in blocks
        for line in block.splitlines()[2:]
    )
    assert dialogue.strip() in normalized_text.replace(".", "")
    assert event in normalized_text
    previous_end = None
    for block in blocks:
        lines = block.splitlines()
        start, end = lines[1].split(" --> ")
        assert len(lines[2:]) <= 2
        assert all(len(line) <= processor.max_chars_per_line for line in lines[2:])
        if previous_end is not None:
            assert start >= previous_end
        previous_end = end


def test_distant_point_tokens_are_not_interpolated_as_one_cluster():
    processor = SrtProcessor({
        "language_code": "eng",
        "words": [
            {"type": "word", "text": "A", "start": 1.0, "end": 1.0},
            {"type": "word", "text": "B", "start": 10.0, "end": 10.0},
            {"type": "word", "text": "C", "start": 11.0, "end": 12.0},
        ],
    })

    assert [(word["start"], word["end"]) for word in processor.words[:2]] == [
        (1.0, 1.0),
        (10.0, 10.0),
    ]


def test_srt_serialization_enforces_one_millisecond_duration():
    processor = SrtProcessor({"language_code": "eng", "words": []})
    srt = processor._generate_final_srt_content([
        {"text": "visible", "start": 21.626, "end": 21.626000000000005}
    ])

    assert "00:00:21,626 --> 00:00:21,627" in srt
