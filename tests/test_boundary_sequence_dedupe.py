from core.async_chunk_processor import AsyncChunkProcessor, dedupe_boundary_words


def _word(text, start, end, speaker_id="speaker_0"):
    return {
        "type": "word",
        "text": text,
        "start": start,
        "end": end,
        "speaker_id": speaker_id,
    }


def _spacing(start):
    return {
        "type": "spacing",
        "text": " ",
        "start": start,
        "end": start,
    }


def test_dedupes_longest_suffix_prefix_sequence_while_ignoring_spacing():
    existing = [
        _word("say", 3.60, 3.90),
        _spacing(3.90),
        _word("one", 4.00, 4.30),
        _spacing(4.30),
        _word("two", 4.40, 4.70),
        _spacing(4.70),
        _word("three", 4.80, 5.10),
    ]
    new = [
        _word("one", 4.03, 4.33),
        _spacing(4.33),
        _word("two", 4.43, 4.73),
        _spacing(4.73),
        _word("three", 4.84, 5.14),
        _spacing(5.14),
        _word("again", 5.20, 5.55),
    ]

    result = dedupe_boundary_words(existing, new)

    assert [item["type"] for item in result] == ["spacing", "word"]
    assert "".join(item["text"] for item in result) == " again"


def test_invalid_non_spacing_item_blocks_prefix_matching():
    existing = [_word("hello", 4.95, 5.02)]
    new = [
        {"type": "word", "text": "unresolved"},
        _word("hello", 5.00, 5.08),
    ]

    assert dedupe_boundary_words(existing, new) == new


def test_does_not_delete_non_contiguous_matches_from_tail_window():
    existing = [
        _word("alpha", 4.0, 4.3),
        _word("beta", 4.3, 4.6),
        _word("gamma", 4.6, 4.9),
    ]
    new = [
        _word("alpha", 4.02, 4.32),
        _word("gamma", 4.62, 4.92),
        _word("next", 5.0, 5.3),
    ]

    assert dedupe_boundary_words(existing, new) == new


def test_keeps_same_text_when_speakers_are_different():
    existing = [_word("yes", 4.90, 5.20, speaker_id="speaker_0")]
    new = [
        _word("yes", 4.91, 5.21, speaker_id="speaker_1"),
        _word("next", 5.30, 5.60, speaker_id="speaker_1"),
    ]

    assert dedupe_boundary_words(existing, new) == new


def test_keeps_legitimate_single_word_repeat_with_only_light_overlap():
    existing = [_word("no", 4.90, 5.20)]
    new = [
        _word("no", 5.15, 5.45),
        _word("next", 5.50, 5.80),
    ]

    assert dedupe_boundary_words(existing, new) == new


def test_dedupes_single_word_only_when_time_alignment_is_confident():
    existing = [_word("hello", 4.95, 5.02)]
    new = [
        _word("hello", 5.00, 5.08),
        _word("world", 5.20, 5.60),
    ]

    result = dedupe_boundary_words(existing, new)

    assert [item["text"] for item in result] == ["world"]


def test_keeps_single_word_when_durations_are_materially_different():
    existing = [_word("no", 4.90, 5.90)]
    new = [_word("no", 5.00, 5.20), _word("next", 5.30, 5.60)]

    assert dedupe_boundary_words(existing, new) == new


def test_keeps_single_zero_duration_repeat():
    existing = [_word("no", 5.0, 5.0)]
    new = [_word("no", 5.0, 5.0), _word("next", 5.1, 5.2)]

    assert dedupe_boundary_words(existing, new) == new


def test_does_not_create_double_spacing_after_dedupe():
    existing = [_word("hello", 4.95, 5.02), _spacing(5.02)]
    new = [
        _word("hello", 5.00, 5.08),
        _spacing(5.08),
        _word("world", 5.20, 5.60),
    ]

    result = dedupe_boundary_words(existing, new)

    assert "".join(item["text"] for item in existing + result) == "hello world"


def test_keeps_repeat_when_only_one_side_has_speaker_id():
    existing = [_word("yes", 4.90, 5.20, speaker_id="speaker_0")]
    new = [
        _word("yes", 4.91, 5.21, speaker_id=None),
        _word("next", 5.30, 5.60, speaker_id=None),
    ]

    assert dedupe_boundary_words(existing, new) == new


def test_async_merge_rebuilds_text_from_deduped_words():
    processor = AsyncChunkProcessor()
    processor.completed_chunks = {
        0: {"text": "hello", "words": [_word("hello", 4.95, 5.02)]},
        1: {
            "text": "hello world",
            "words": [
                _word("hello", 5.00, 5.08),
                _spacing(5.08),
                _word("world", 5.20, 5.60),
            ],
        },
    }

    merged = processor._merge_transcripts()

    assert merged["text"] == "hello world"
    assert merged["text"] == "".join(word["text"] for word in merged["words"])
