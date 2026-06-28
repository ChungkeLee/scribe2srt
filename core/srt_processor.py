#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SRT字幕处理器模块
使用新的两阶段算法：基于标点符号的句子预分割 + 智能合并
"""

import math
import re
from typing import Dict, List

from .config import (
    MIN_SUBTITLE_DURATION, MIN_SUBTITLE_GAP, CPS_SETTINGS, CPL_SETTINGS
)
from .sentence_splitter import SentenceSplitter
from .intelligent_merger import IntelligentMerger
from .punctuation_handler import PunctuationHandler
from .language_utils import is_cjk_language


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


class SrtProcessor:
    """
    SRT字幕处理器类
    
    核心功能：
    - 使用新的两阶段算法处理转录数据
    - 基于标点符号的语义分割
    - 智能合并优化
    - 多语言支持和专业标准遵循
    """
    
    def __init__(self, json_data: Dict, max_subtitle_duration: float = None,
                 subtitle_settings: Dict = None):
        self.srt_content = []
        self.line_number = 1
        self.language = json_data.get("language_code", "eng")[:3] # e.g., "eng"
        self.is_cjk = is_cjk_language(self.language)

        # 如果提供了高级设置，使用它们；否则使用默认值
        if subtitle_settings:
            self.max_subtitle_duration = subtitle_settings.get("max_subtitle_duration", 7.0)
            self.min_subtitle_duration = subtitle_settings.get("min_subtitle_duration", MIN_SUBTITLE_DURATION)
            self.min_subtitle_gap = subtitle_settings.get("min_subtitle_gap", MIN_SUBTITLE_GAP)

            # 使用用户自定义的CPS和CPL设置
            if self.is_cjk:
                self.max_cps = subtitle_settings.get("cjk_cps", CPS_SETTINGS["cjk"])
                self.max_chars_per_line = subtitle_settings.get("cjk_chars_per_line", CPL_SETTINGS["cjk"])
            else:
                self.max_cps = subtitle_settings.get("latin_cps", CPS_SETTINGS["latin"])
                self.max_chars_per_line = subtitle_settings.get("latin_chars_per_line", CPL_SETTINGS["latin"])
        else:
            # 使用传入参数或默认值（向后兼容）
            self.max_subtitle_duration = max_subtitle_duration if max_subtitle_duration is not None else 7.0
            self.min_subtitle_duration = MIN_SUBTITLE_DURATION
            self.min_subtitle_gap = MIN_SUBTITLE_GAP

            # 根据语言动态设置参数
            self.max_chars_per_line = self._get_max_chars_for_language()
            self.max_cps = self._get_max_cps_for_language()

        self._preprocess_words(json_data)

    def _get_max_chars_for_language(self) -> int:
        """Returns the recommended max characters per line based on language."""
        if self.is_cjk:
            return CPL_SETTINGS["cjk"]
        else:
            return CPL_SETTINGS["latin"]

    def _get_max_cps_for_language(self) -> float:
        """Returns the recommended max CPS based on language."""
        if self.is_cjk:
            return CPS_SETTINGS["cjk"]
        else:
            return CPS_SETTINGS["latin"]

    def _get_dynamic_cps_limit(self, text: str) -> float:
        """
        返回当前语言的CPS硬限制。

        Args:
            text: 文本内容

        Returns:
            当前语言的CPS限制
        """
        return self.max_cps

    def _preprocess_words(self, json_data: Dict):
        """
        Pre-processes the word list to handle language-specific quirks,
        such as merging standalone CJK punctuation and filtering out spacing characters.
        Also separates audio_event types for independent processing.
        """
        raw_words = json_data.get("words", [])
        self.words = []
        self.audio_events = []  # 独立存储音频事件

        for word_info in raw_words:
            # 首先检查是否为音频事件类型
            if word_info.get('type') == 'audio_event':
                self.audio_events.append(word_info.copy())
                continue

            # Skip spacing characters to fix timing issues with Latin text
            # But preserve the space character in the text of the previous word
            if word_info.get('type') == 'spacing':
                # Add space to the previous word if it exists and doesn't already end with space
                if (self.words and
                    word_info.get('text', '').strip() == '' and  # Only for actual spaces
                    self.words[-1].get('type') == 'word' and
                    not self.words[-1]['text'].endswith(' ')):
                    self.words[-1]['text'] += ' '
                continue

            is_cjk_punctuation = len(word_info['text']) == 1 and word_info['text'] in "。？！」「、・，"
            if is_cjk_punctuation and self.words:
                prev_word = self.words[-1]
                if prev_word.get("type") == "word" and prev_word['text'] and prev_word['text'][-1] not in "。？！」「、・，":
                    prev_word['text'] += word_info['text']
                    prev_word['end'] = word_info['end']
                    continue
            self.words.append(word_info)

    def _create_audio_event_entries(self) -> List[Dict]:
        """
        为音频事件创建独立的字幕条目

        Returns:
            音频事件字幕条目列表
        """
        audio_event_entries = []

        for event in self.audio_events:
            # 创建音频事件字幕条目
            entry = {
                'text': event['text'],
                'start': event['start'],
                'end': event['end'],
                'words': [event],
                'is_audio_event': True,
                'word_count': 0,  # 音频事件不计入单词数
                'char_count': len(event['text'].replace(' ', ''))
            }
            audio_event_entries.append(entry)

        return audio_event_entries

    def create_srt(self) -> str:
        """
        Creates the full SRT content using the new two-stage approach:
        1. Sentence-level pre-splitting based on punctuation priority (only for word types)
        2. Independent audio_event processing
        3. Intelligent merging based on CPS, CPL, and display time rules
        4. Final integration and sorting
        """
        if not self.words and not self.audio_events:
            return ""

        # Stage 1: Sentence-level pre-splitting (only for word types)
        basic_entries = []

        if self.words:
            sentence_splitter = SentenceSplitter(self.language)
            sentence_groups = sentence_splitter.split_into_sentence_groups(self.words)
            basic_entries = sentence_splitter.create_basic_subtitle_entries(sentence_groups)

        # Stage 2: Independent audio_event processing
        audio_event_entries = self._create_audio_event_entries()

        # Stage 3: Intelligent merging (only for word-based entries)
        merged_entries = []
        if basic_entries:
            subtitle_settings = {
                'min_subtitle_duration': self.min_subtitle_duration,
                'min_subtitle_gap': self.min_subtitle_gap,
                'max_subtitle_duration': self.max_subtitle_duration,
                'cjk_cps': self.max_cps if self.is_cjk else CPS_SETTINGS["cjk"],
                'latin_cps': self.max_cps if not self.is_cjk else CPS_SETTINGS["latin"],
                'cjk_chars_per_line': self.max_chars_per_line if self.is_cjk else CPL_SETTINGS["cjk"],
                'latin_chars_per_line': self.max_chars_per_line if not self.is_cjk else CPL_SETTINGS["latin"]
            }

            intelligent_merger = IntelligentMerger(self.language, subtitle_settings)
            merged_entries = intelligent_merger.merge_basic_entries(basic_entries)
            merged_entries = intelligent_merger.optimize_merged_entries(merged_entries)
            merged_entries = self._split_entries_by_constraints(merged_entries)
            merged_entries = self._merge_short_entries(merged_entries)

        # Stage 4: Combine and sort all entries (word-based + audio events)
        all_entries = merged_entries + audio_event_entries
        all_entries.sort(key=lambda x: x['start'])  # 按时间顺序排序
        all_entries = [self._apply_terminal_punctuation(entry) for entry in all_entries]
        all_entries = self._split_entries_by_constraints(all_entries)
        all_entries = [self._apply_terminal_punctuation(entry) for entry in all_entries]
        all_entries = self._normalize_timeline(all_entries)

        # Stage 5: Generate final SRT content with optimized display formatting
        return self._generate_final_srt_content(all_entries)

    def _split_entries_by_constraints(self, entries: List[Dict]) -> List[Dict]:
        """Split word-based entries that cannot fit subtitle timing or display rules."""
        split_entries = []
        for entry in entries:
            split_entries.extend(self._split_entry_if_needed(entry))
        return split_entries

    def _split_entry_if_needed(self, entry: Dict, depth: int = 0) -> List[Dict]:
        if entry.get('is_audio_event', False) or self._entry_is_compliant(entry):
            return [entry]

        words = [word for word in entry.get('words', []) if word.get('type') == 'word']
        if len(words) <= 1 or depth >= 12:
            return self._split_single_word_entry(entry) if words else [entry]

        split_index = self._find_best_word_split(words)
        if split_index is None:
            return [self._build_entry_from_words(words, entry)]

        left = self._build_entry_from_words(words[:split_index], entry)
        right = self._build_entry_from_words(words[split_index:], entry)

        return (
            self._split_entry_if_needed(left, depth + 1) +
            self._split_entry_if_needed(right, depth + 1)
        )

    def _entry_is_compliant(self, entry: Dict) -> bool:
        text = entry.get('text', '').strip()
        if not text:
            return True

        start, end = self._entry_time_bounds(entry)
        duration = end - start
        if duration <= 0 or duration > self.max_subtitle_duration:
            return False

        if self._required_duration_for_cps(text) > self.max_subtitle_duration:
            return False

        display_lines = self._wrap_text_unlimited(text)
        if len(display_lines) > 2:
            return False
        if any(len(line) > self.max_chars_per_line for line in display_lines):
            return False

        return True

    def _split_single_word_entry(self, entry: Dict) -> List[Dict]:
        """Split a long single timed token/phrase proportionally by text."""
        words = [word for word in entry.get('words', []) if word.get('type') == 'word']
        if not words:
            return [entry]

        text = entry.get('text', '').strip()
        if not text:
            return [entry]

        start, end = self._entry_time_bounds(entry)
        duration = max(0.001, end - start)
        max_chars_per_subtitle = max(1, self.max_chars_per_line * 2)
        readable_chars = max(1, self._count_readable_chars(text))

        part_count = max(
            1,
            math.ceil(duration / self.max_subtitle_duration),
            math.ceil(readable_chars / max_chars_per_subtitle)
        )

        if part_count <= 1:
            return [self._build_entry_from_words(words, entry)]

        pieces = self._split_text_into_balanced_pieces(text, part_count)
        if len(pieces) <= 1:
            pieces = [text]

        display_duration = min(duration, len(pieces) * self.max_subtitle_duration)
        piece_duration = max(0.001, display_duration / len(pieces))
        cursor = start
        split_entries = []

        for index, piece in enumerate(pieces):
            if index == len(pieces) - 1 and display_duration >= duration:
                piece_end = end
            else:
                piece_end = min(end, cursor + piece_duration)

            source_word = words[0].copy()
            source_word['text'] = piece
            source_word['start'] = round(cursor, 3)
            source_word['end'] = round(piece_end, 3)

            split_entries.append({
                'text': piece,
                'start': source_word['start'],
                'end': source_word['end'],
                'words': [source_word],
                'is_audio_event': entry.get('is_audio_event', False),
                'word_count': 1,
                'char_count': self._count_readable_chars(piece)
            })
            cursor = piece_end

        return split_entries

    def _split_text_into_balanced_pieces(self, text: str, part_count: int) -> List[str]:
        remaining = text.strip()
        pieces = []

        for index in range(part_count - 1):
            remaining_parts = part_count - index
            target_length = max(1, math.ceil(len(remaining) / remaining_parts))
            split_pos = self._find_balanced_text_split(remaining, target_length)

            piece = remaining[:split_pos].strip()
            if piece:
                self._append_text_piece(pieces, piece)
            remaining = remaining[split_pos:].strip()

            if not remaining:
                break

        if remaining:
            self._append_text_piece(pieces, remaining)

        return pieces

    def _find_balanced_text_split(self, text: str, target_length: int) -> int:
        if len(text) <= 1:
            return len(text)

        fallback = min(max(1, target_length), len(text) - 1)
        min_pos = max(1, int(target_length * 0.65))
        max_pos = min(len(text) - 1, max(min_pos, int(target_length * 1.35)))

        for pos in range(max_pos, min_pos - 1, -1):
            prev_char = text[pos - 1]
            current_char = text[pos] if pos < len(text) else ""
            if prev_char in PunctuationHandler.ALL_PUNCTUATION:
                return pos
            if current_char == " ":
                return pos

        return fallback

    def _append_text_piece(self, pieces: List[str], piece: str):
        if not piece:
            return
        if pieces and all(char in PunctuationHandler.ALL_PUNCTUATION for char in piece):
            pieces[-1] += piece
            return
        pieces.append(piece)

    def _build_entry_from_words(self, words: List[Dict], template: Dict = None) -> Dict:
        template = template or {}
        actual_words = [word for word in words if word.get('type') == 'word']
        if not actual_words:
            return template.copy()

        text = ''.join(word.get('text', '') for word in words).strip()
        if not self.is_cjk:
            text = re.sub(r'\s+', ' ', text)

        return {
            'text': text,
            'start': actual_words[0]['start'],
            'end': actual_words[-1]['end'],
            'words': [word.copy() for word in words],
            'is_audio_event': template.get('is_audio_event', False),
            'word_count': len(actual_words),
            'char_count': self._count_readable_chars(text)
        }

    def _find_best_word_split(self, words: List[Dict]) -> int:
        if len(words) < 2:
            return None

        total_text = ''.join(word.get('text', '') for word in words).strip()
        total_chars = max(1, self._count_readable_chars(total_text))
        best_index = None
        best_score = float('inf')

        for index in range(1, len(words)):
            left_words = words[:index]
            right_words = words[index:]
            left_text = ''.join(word.get('text', '') for word in left_words).strip()
            right_text = ''.join(word.get('text', '') for word in right_words).strip()
            left_chars = self._count_readable_chars(left_text)
            right_chars = self._count_readable_chars(right_text)

            if left_chars == 0 or right_chars == 0:
                continue

            balance_penalty = abs(left_chars - right_chars) / total_chars
            score = balance_penalty * 10

            previous_text = left_words[-1].get('text', '').strip()
            has_punct, _, priority = PunctuationHandler.word_ends_with_punctuation(previous_text)
            if has_punct:
                score -= {0: 60, 1: 35, 2: 15}.get(priority, 0)

            gap = right_words[0].get('start', left_words[-1].get('end', 0)) - left_words[-1].get('end', 0)
            if gap > 0:
                score -= min(gap, 1.0) * 2

            score += self._constraint_penalty(left_text, left_words)
            score += self._constraint_penalty(right_text, right_words)

            if left_chars < 4:
                score += 2
            if right_chars < 4:
                score += 2

            if score < best_score:
                best_score = score
                best_index = index

        return best_index

    def _constraint_penalty(self, text: str, words: List[Dict]) -> float:
        duration = words[-1].get('end', 0) - words[0].get('start', 0)
        effective_duration = max(duration, self.min_subtitle_duration)
        lines = self._wrap_text_unlimited(text)
        penalty = max(0, len(lines) - 2) * 8
        penalty += sum(max(0, len(line) - self.max_chars_per_line) for line in lines) / 5

        cps = self._calculate_cps(text, effective_duration)
        cps_limit = self._get_dynamic_cps_limit(text)
        if cps > cps_limit:
            penalty += (cps - cps_limit) / max(cps_limit, 1)

        if duration <= 0:
            penalty += 25

        return penalty

    def _normalize_timeline(self, entries: List[Dict]) -> List[Dict]:
        """Keep subtitles synchronized to source word times without cumulative drift."""
        if not entries:
            return []

        sorted_entries = sorted((entry.copy() for entry in entries), key=lambda x: x['start'])
        normalized = []

        for entry in sorted_entries:
            current = entry.copy()
            is_audio_event = current.get('is_audio_event', False)
            source_start, source_end = self._entry_time_bounds(current)
            coverage_duration = max(0.001, source_end - source_start)

            target_duration = self._target_display_duration(current, coverage_duration, is_audio_event)
            current['start'], current['end'] = self._preferred_timing_for_entry(
                source_start,
                source_end,
                target_duration,
                coverage_duration,
                is_audio_event
            )

            if normalized:
                self._resolve_timing_conflict(normalized[-1], current, source_start)

            if current['end'] <= current['start']:
                current['end'] = current['start'] + 0.001

            normalized.append(current)

        return normalized

    def _target_display_duration(self, entry: Dict, coverage_duration: float, is_audio_event: bool) -> float:
        if is_audio_event and coverage_duration > self.max_subtitle_duration:
            return self.max_subtitle_duration

        target_duration = max(
            coverage_duration,
            self.min_subtitle_duration,
            self._required_duration_for_cps(entry.get('text', ''))
        )

        if coverage_duration <= self.max_subtitle_duration:
            return min(target_duration, self.max_subtitle_duration)

        return coverage_duration if not is_audio_event else self.max_subtitle_duration

    def _preferred_timing_for_entry(self, source_start: float, source_end: float,
                                    target_duration: float, coverage_duration: float,
                                    is_audio_event: bool) -> tuple:
        if is_audio_event:
            return source_start, source_start + target_duration

        if coverage_duration > self.max_subtitle_duration:
            return source_start, source_start + self.max_subtitle_duration

        start = max(0.0, source_end - target_duration)
        end = max(source_end, start + target_duration)
        return start, end

    def _resolve_timing_conflict(self, previous: Dict, current: Dict, current_source_start: float):
        previous_source_start, previous_source_end = self._entry_time_bounds(previous)
        source_gap = max(0.0, current_source_start - previous_source_end)
        desired_gap = min(self.min_subtitle_gap, source_gap)

        if previous['end'] + desired_gap <= current['start']:
            return

        if current['start'] < current_source_start:
            current['start'] = current_source_start
            if current['end'] <= current['start']:
                current['end'] = current['start'] + 0.001

        latest_previous_end = current['start'] - desired_gap
        minimum_previous_end = self._minimum_timing_end(previous)

        if previous['end'] > latest_previous_end and latest_previous_end >= minimum_previous_end:
            previous['end'] = latest_previous_end

        if previous['end'] > current['start']:
            latest_previous_end = current['start'] - 0.001
            if latest_previous_end >= minimum_previous_end:
                previous['end'] = latest_previous_end
            else:
                current['start'] = previous['end'] + 0.001
                if current['end'] <= current['start']:
                    current['end'] = current['start'] + 0.001

    def _minimum_timing_end(self, entry: Dict) -> float:
        if entry.get('is_audio_event', False):
            return entry['start'] + min(self.min_subtitle_duration, self.max_subtitle_duration)

        _, source_end = self._entry_time_bounds(entry)
        return max(source_end, entry['start'] + 0.001)

    def _entry_time_bounds(self, entry: Dict) -> tuple:
        timed_items = [
            item for item in entry.get('words', [])
            if isinstance(item.get('start'), (int, float)) and isinstance(item.get('end'), (int, float))
        ]
        if timed_items:
            return (
                min(item['start'] for item in timed_items),
                max(item['end'] for item in timed_items)
            )

        return entry.get('start', 0), entry.get('end', 0)

    def _apply_terminal_punctuation(self, entry: Dict) -> Dict:
        normalized_entry = entry.copy()
        normalized_entry['text'] = self._ensure_terminal_punctuation(
            normalized_entry.get('text', '')
        )
        normalized_entry['char_count'] = self._count_readable_chars(normalized_entry['text'])
        return normalized_entry

    def _ensure_terminal_punctuation(self, text: str) -> str:
        text = (text or '').strip()
        if not text:
            return text

        if self._has_acceptable_terminal_punctuation(text):
            return text

        terminal = self._default_terminal_punctuation()
        closers = PunctuationHandler.TRAILING_CLOSERS
        suffix = ""
        base = text

        while base and base[-1] in closers:
            suffix = base[-1] + suffix
            base = base[:-1].rstrip()

        if not base:
            return text + terminal

        if self._has_acceptable_terminal_punctuation(base):
            return base + suffix

        return base + terminal + suffix

    def _default_terminal_punctuation(self) -> str:
        if self.language in ["zho", "chi", "zh", "jpn", "ja"]:
            return "。"
        return "."

    def _has_acceptable_terminal_punctuation(self, text: str) -> bool:
        text = (text or '').strip()
        if not text:
            return False

        unacceptable_endings = {"-", "(", "[", "{", "（", "「", "【", "《"}
        if text[-1] in unacceptable_endings:
            return False

        return PunctuationHandler.word_ends_with_punctuation(text)[0]

    def _merge_short_entries(self, entries: List[Dict]) -> List[Dict]:
        if not entries:
            return []

        sorted_entries = sorted(entries, key=lambda x: x['start'])
        merged = []
        index = 0

        while index < len(sorted_entries):
            current = sorted_entries[index].copy()

            while index + 1 < len(sorted_entries):
                next_entry = sorted_entries[index + 1]
                if not self._can_merge_short_entries(current, next_entry):
                    break

                current = self._merge_two_post_entries(current, next_entry)
                index += 1

            merged.append(current)
            index += 1

        return merged

    def _can_merge_short_entries(self, entry1: Dict, entry2: Dict) -> bool:
        if entry1.get('is_audio_event', False) or entry2.get('is_audio_event', False):
            return False

        duration1 = entry1['end'] - entry1['start']
        duration2 = entry2['end'] - entry2['start']
        if duration1 >= self.min_subtitle_duration and duration2 >= self.min_subtitle_duration:
            return False

        gap = entry2['start'] - entry1['end']
        if gap < 0 or gap > 0.5:
            return False

        merged_text = self._join_text(entry1.get('text', ''), entry2.get('text', ''))
        merged_duration = entry2['end'] - entry1['start']
        if merged_duration > self.max_subtitle_duration:
            return False

        lines = self._wrap_text_unlimited(merged_text)
        if len(lines) > 2:
            return False
        if any(len(line) > self.max_chars_per_line for line in lines):
            return False

        return True

    def _merge_two_post_entries(self, entry1: Dict, entry2: Dict) -> Dict:
        text = self._join_text(entry1.get('text', ''), entry2.get('text', ''))
        words = entry1.get('words', []) + entry2.get('words', [])

        return {
            'text': text,
            'start': entry1['start'],
            'end': entry2['end'],
            'words': words,
            'is_audio_event': False,
            'word_count': entry1.get('word_count', 0) + entry2.get('word_count', 0),
            'char_count': self._count_readable_chars(text)
        }

    def _join_text(self, text1: str, text2: str) -> str:
        text1 = text1.strip()
        text2 = text2.strip()

        if not text1:
            return text2
        if not text2:
            return text1
        if self.is_cjk:
            return text1 + text2
        return f"{text1} {text2}"

    def _calculate_cps(self, text: str, duration: float) -> float:
        if duration <= 0:
            return float('inf')
        return self._count_readable_chars(text) / duration

    def _count_readable_chars(self, text: str) -> int:
        return len(re.sub(r'\s+', '', text or ''))

    def _required_duration_for_cps(self, text: str) -> float:
        char_count = self._count_readable_chars(text)
        if char_count == 0:
            return 0.0
        # Add a small millisecond margin because SRT serialization rounds times.
        return (char_count / self._get_dynamic_cps_limit(text)) + 0.005

    def _wrap_text_unlimited(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        lines = []
        remaining_text = text

        while remaining_text:
            if len(remaining_text) <= self.max_chars_per_line:
                lines.append(remaining_text)
                break

            split_pos = PunctuationHandler.find_split_position(
                remaining_text,
                self.max_chars_per_line
            )
            lines.append(remaining_text[:split_pos].strip())
            remaining_text = remaining_text[split_pos:].strip()

        return [line for line in lines if line]

    def _generate_final_srt_content(self, entries: List[Dict]) -> str:
        """
        Generate final SRT content with optimized display formatting
        
        Args:
            entries: List of optimized subtitle entries
            
        Returns:
            Final SRT content string
        """
        if not entries:
            return ""
        
        srt_lines = []
        
        for i, entry in enumerate(entries, 1):
            # Format timing
            start_time_str = format_srt_time(entry['start'])
            end_time_str = format_srt_time(entry['end'])
            
            # Optimize text display format
            formatted_text = self._optimize_text_display(entry['text'])
            
            # Generate SRT entry
            srt_entry = f"{i}\n{start_time_str} --> {end_time_str}\n{formatted_text}\n"
            srt_lines.append(srt_entry)
        
        return "\n".join(srt_lines)
    
    def _optimize_text_display(self, text: str) -> str:
        """
        Optimize text display format: prioritize single line, break at punctuation if needed
        
        Args:
            text: Original text
            
        Returns:
            Optimized display text
        """
        text = text.strip()
        if not text:
            return text
        
        # If text fits in single line, return as-is
        if len(text) <= self.max_chars_per_line:
            return text
        
        # Need to split into multiple lines, prioritize punctuation breaks
        return self._split_text_into_lines(text)

    def _split_text_into_lines(self, text: str) -> str:
        """
        Intelligently splits a text block into a maximum of two lines,
        following professional subtitle standards for line breaking.

        Prioritizes semantic completeness over visual aesthetics (Netflix standard).
        """
        text = text.strip()
        if len(text) <= self.max_chars_per_line:
            return text

        lines = self._wrap_text_unlimited(text)
        if len(lines) <= 2:
            return "\n".join(lines)

        # This is a last-resort display fallback. Structural splitting should
        # normally keep entries within two lines before this point.
        return "\n".join([lines[0], " ".join(lines[1:])])




def create_srt_from_json(json_data: Dict, max_subtitle_duration: float = None,
                        subtitle_settings: Dict = None) -> str:
    """
    Processes transcription JSON data to create a professional SRT file.

    Args:
        json_data: Transcription data from ElevenLabs or similar service
        max_subtitle_duration: Maximum duration for a single subtitle (default: 7.0s)
        subtitle_settings: Dictionary containing advanced subtitle settings

    Returns:
        Professional SRT content following industry standards
    """
    processor = SrtProcessor(json_data, max_subtitle_duration, subtitle_settings)
    return processor.create_srt()
