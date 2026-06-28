# -*- coding: utf-8 -*-

"""
这个文件定义了在后台线程中执行所有处理任务的 Worker 类。
"""

import os
import sys
import json
import re
import shutil
import subprocess
import tempfile
from typing import Optional, List, Dict, Any, Tuple

from PySide6.QtCore import QObject, Signal, QThreadPool

from api.client import ElevenLabsSTTClient
from .srt_processor import create_srt_from_json
from .async_chunk_processor import AsyncChunkProcessor
from .ffmpeg_utils import get_media_info

class Worker(QObject):
    """
    在单独的线程中处理文件转录、切分和SRT生成任务。
    """
    finished = Signal(str)
    error = Signal(str)
    log_message = Signal(str)
    progress_updated = Signal(int, int)
    chunk_progress = Signal(int, str, str)  # chunk_index, status, message
    chunks_ready = Signal(list)  # chunk_paths - 通知UI设置分段进度条

    def __init__(self, file_path: str, language_code: str, tag_audio_events: bool,
                 max_subtitle_duration: float, split_duration_min: int,
                 original_file_path: Optional[str] = None, ffmpeg_available: bool = False,
                 restore_state: Optional[Dict[str, Any]] = None, subtitle_settings: Optional[Dict] = None,
                 enable_async_processing: bool = True, max_concurrent_chunks: int = 3,
                 max_retries: int = 3, api_rate_limit_per_minute: int = 30):
        super().__init__()
        self.file_path = file_path
        self.original_file_path = original_file_path if original_file_path else file_path
        self.language_code = language_code
        self.tag_audio_events = tag_audio_events
        self.max_subtitle_duration = max_subtitle_duration
        self.split_duration_sec = split_duration_min * 60
        self.ffmpeg_available = ffmpeg_available
        self.restore_state = restore_state
        self.subtitle_settings = subtitle_settings
        
        self.uploader = None
        self.client = ElevenLabsSTTClient(signals_forwarder=self, ffmpeg_available=self.ffmpeg_available)

        # 异步片段处理器
        self.async_processor = None
        self.async_base_chunk_index = 0
        # 从恢复状态或使用传入参数配置异步处理
        if self.restore_state:
            self.enable_async_processing = self.restore_state.get("enable_async_processing", enable_async_processing)
            self.max_concurrent_chunks = self.restore_state.get("max_concurrent_chunks", max_concurrent_chunks)
            self.max_retries = self.restore_state.get("max_retries", max_retries)
            self.api_rate_limit_per_minute = self.restore_state.get("api_rate_limit_per_minute", api_rate_limit_per_minute)
        else:
            self.enable_async_processing = enable_async_processing
            self.max_concurrent_chunks = max_concurrent_chunks
            self.max_retries = max_retries
            self.api_rate_limit_per_minute = api_rate_limit_per_minute

        self._is_cancelled = False
        
        if self.restore_state:
            self.temp_chunks = self.restore_state.get("temp_chunks", [])
            self.owned_temp_chunks = self.restore_state.get("owned_temp_chunks", [])
            self.combined_transcript = self.restore_state.get("combined_transcript", {})
            self.current_chunk_index = self.restore_state.get("current_chunk_index", 0)
            self.total_chunks = self.restore_state.get("total_chunks", 0)
            self.time_offset = self.restore_state.get("time_offset", 0.0)
            self.chunk_offsets = self.restore_state.get("chunk_offsets", [])
            self.temp_chunk_dir = self.restore_state.get("temp_chunk_dir")
            # 恢复处理模式信息
            self.was_single_file_mode = self.restore_state.get("was_single_file_mode", False)
            self.extracted_audio_file = self.restore_state.get("extracted_audio_file", None)
        else:
            self.temp_chunks = []
            self.owned_temp_chunks = []
            self.combined_transcript = {}
            self.current_chunk_index = 0
            self.total_chunks = 0
            self.time_offset = 0.0
            self.chunk_offsets = []
            self.temp_chunk_dir = None
            self.was_single_file_mode = False
            self.extracted_audio_file = None

        self._cancellation_reported = False

    def get_state(self) -> Dict[str, Any]:
        """获取当前worker的状态，用于任务恢复。"""
        # 获取异步处理器的进度信息
        async_progress = {}
        restorable_chunk_index = self.current_chunk_index
        if self.async_processor:
            async_progress = self.async_processor.get_progress_info()
            # 异步处理时 current_chunk_index 会被上传进度用于 UI 显示，
            # 不能作为可恢复的连续完成位置。恢复时从本轮异步处理的起点重跑，
            # 保证失败重试不跳过尚未合并进 combined_transcript 的片段。
            restorable_chunk_index = self.async_base_chunk_index

        return {
            "file_path": self.file_path,  # 添加 file_path 到状态中
            "temp_chunks": self.temp_chunks,
            "owned_temp_chunks": self.owned_temp_chunks,
            "chunk_offsets": self.chunk_offsets,
            "temp_chunk_dir": self.temp_chunk_dir,
            "combined_transcript": self.combined_transcript,
            "current_chunk_index": restorable_chunk_index,
            "total_chunks": self.total_chunks,
            "time_offset": self.time_offset,
            "original_file_path": self.original_file_path,
            "language_code": self.language_code,
            "tag_audio_events": self.tag_audio_events,
            "max_subtitle_duration": self.max_subtitle_duration,
            "split_duration_min": self.split_duration_sec / 60,
            "ffmpeg_available": self.ffmpeg_available,
            # 异步处理相关状态
            "enable_async_processing": self.enable_async_processing,
            "max_concurrent_chunks": self.max_concurrent_chunks,
            "max_retries": self.max_retries,
            "api_rate_limit_per_minute": self.api_rate_limit_per_minute,
            "async_progress": async_progress,
            # 添加处理模式信息，用于正确的重试逻辑
            "was_single_file_mode": self.total_chunks == 1,
            "extracted_audio_file": getattr(self, 'extracted_audio_file', None),
        }

    def run(self):
        """任务执行的入口点。"""
        is_restoring = self.restore_state and self.restore_state.get("temp_chunks")

        if is_restoring:
            self.log_message.emit("...从断点处恢复任务...")

            # 检查临时文件是否存在
            missing_files = [p for p in self.temp_chunks if not os.path.exists(p)]

            if missing_files:
                if self.was_single_file_mode:
                    # 单文件模式：重新提取音频而不是切分
                    self.log_message.emit("检测到提取的音频文件丢失，正在重新提取...")

                    # 重新执行音频提取逻辑
                    original_file = self.restore_state.get("original_file_path", self.original_file_path)

                    # 检查是否需要从视频提取音频
                    _, ext = os.path.splitext(original_file)
                    video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.flv', '.webm']

                    if ext.lower() in video_extensions and self.ffmpeg_available:
                        # 重新提取音频
                        from core.ffmpeg_utils import get_media_info, extract_audio
                        from ui.main_window import CODEC_EXTENSION_MAP, DEFAULT_AUDIO_EXTENSION

                        media_info = get_media_info(original_file)
                        codec = media_info.get("codec") if media_info else None
                        extension = CODEC_EXTENSION_MAP.get(codec, DEFAULT_AUDIO_EXTENSION) if codec else DEFAULT_AUDIO_EXTENSION

                        base_name, _ = os.path.splitext(os.path.basename(original_file))
                        temp_audio_path = os.path.join(os.path.dirname(original_file), f"temp_audio_{base_name}{extension}")

                        if not extract_audio(original_file, temp_audio_path):
                            self.error.emit("恢复任务失败：无法重新提取音频。")
                            return

                        # 更新文件路径
                        self.file_path = temp_audio_path
                        self.extracted_audio_file = temp_audio_path
                        self.temp_chunks = [temp_audio_path]
                        self.chunk_offsets = [0.0]
                        self.log_message.emit(f"音频重新提取完成: {os.path.basename(temp_audio_path)}")
                    else:
                        # 直接使用原始音频文件
                        self.temp_chunks = [original_file]
                        self.chunk_offsets = [0.0]
                        self.log_message.emit("使用原始音频文件继续处理")
                else:
                    # 多片段模式：重新切分音频
                    self.log_message.emit("检测到临时切片文件丢失，正在重新切分...")
                    if not self._split_audio(self.restore_state.get("original_file_path", self.original_file_path)):
                        self.error.emit("恢复任务失败：无法重新切分音频。")
                        return

            self._ensure_chunk_offsets()
            # 恢复模式下的处理逻辑
            self._process_restored_chunks()
            return

        self.log_message.emit("="*50)
        self.log_message.emit(f"开始处理文件: {os.path.basename(self.original_file_path)}")

        media_info = self.client.log_media_info(self.file_path)
        duration = media_info.get("duration") if media_info else 0

        if duration > self.split_duration_sec and self.ffmpeg_available:
            self.log_message.emit(f"文件时长超过 {self.split_duration_sec / 60:.0f} 分钟，将执行自动切分。")
            if self._split_audio(self.file_path):
                self._process_next_chunk()
            else:
                return
        else:
            self.log_message.emit("文件无需切分，执行单文件处理流程。")
            self.total_chunks = 1
            self.temp_chunks.append(self.file_path)
            self.chunk_offsets = [0.0]
            # 记录单文件模式和提取的音频文件信息
            self.was_single_file_mode = True
            if self.file_path != self.original_file_path:
                # 如果处理的文件不是原始文件，说明是提取的音频
                self.extracted_audio_file = self.file_path
            self._process_next_chunk()

    def _split_audio(self, audio_path: str) -> bool:
        """使用 FFmpeg 切分音频文件，并记录每个分片的真实全局起点。"""
        self.log_message.emit("正在切分音频文件...")
        self.chunk_progress.emit(-1, "splitting", "正在切分音频...")

        base_dir = os.path.dirname(os.path.abspath(audio_path)) or os.getcwd()
        base_name, _ = os.path.splitext(os.path.basename(audio_path))

        try:
            self._cleanup_owned_chunk_artifacts()

            media_info = get_media_info(audio_path, lambda message: self.log_message.emit(message))
            duration = media_info.get("duration") if media_info else None
            if not duration or duration <= 0:
                raise RuntimeError("无法获取音频时长，不能安全切分。")

            self.temp_chunk_dir = tempfile.mkdtemp(prefix=f"{base_name}_chunks_", dir=base_dir)

            silence_ranges = self._detect_silence_ranges(audio_path, duration)
            split_points = self._calculate_smart_split_points(duration, silence_ranges)
            segment_ranges = self._build_segment_ranges(duration, split_points)

            if silence_ranges:
                self.log_message.emit(
                    f"检测到 {len(silence_ranges)} 段静音，优先在静音附近切分。"
                )
            else:
                self.log_message.emit("未检测到可用静音点，将使用固定时间切分。")

            chunk_extension = self._chunk_extension_for_audio(audio_path)
            self.log_message.emit(
                f"分片将使用原始音频编码流复制，输出容器: {chunk_extension}"
            )

            self.owned_temp_chunks = []
            self.temp_chunks = []
            self.chunk_offsets = []

            for index, (start, end) in enumerate(segment_ranges):
                chunk_path = os.path.join(
                    self.temp_chunk_dir,
                    f"{base_name}_chunk_{index:03d}{chunk_extension}"
                )
                self._export_audio_segment(audio_path, chunk_path, start, end)
                self.owned_temp_chunks.append(chunk_path)
                self.temp_chunks.append(chunk_path)
                self.chunk_offsets.append(round(start, 3))
                self.log_message.emit(
                    f"片段 {index + 1}: {start:.3f}s -> {end:.3f}s "
                    f"(时长 {end - start:.3f}s)"
                )

            if not self.owned_temp_chunks:
                raise RuntimeError("FFmpeg 执行完毕但未找到任何切分文件。")

            self.total_chunks = len(self.temp_chunks)
            self.log_message.emit(f"成功切分为 {self.total_chunks} 个片段。")
            self.log_message.emit(
                "分片时间偏移: " +
                ", ".join(f"{offset:.3f}s" for offset in self.chunk_offsets)
            )
            # 通知UI设置分段进度条
            self.chunks_ready.emit(self.temp_chunks)
            return True

        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
            error_message = f"音频切分失败: {e}"
            if hasattr(e, 'stderr'):
                error_message += f"\nFFmpeg 输出:\n{e.stderr.strip()}"
            self._cleanup_owned_chunk_artifacts()
            self.error.emit(error_message)
            return False

    def _detect_silence_ranges(self, audio_path: str, duration: float) -> List[Tuple[float, float]]:
        """Detect silence ranges with FFmpeg silencedetect; returns seconds."""
        command = [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", audio_path,
            "-af", "silencedetect=noise=-40dB:d=0.5",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                startupinfo=self._startupinfo()
            )
        except FileNotFoundError:
            return []

        if result.returncode != 0:
            self.log_message.emit("静音检测失败，将回退为固定时间切分。")
            return []

        silence_ranges = []
        current_start = None
        start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
        end_pattern = re.compile(r"silence_end:\s*([0-9.]+)")

        for line in result.stderr.splitlines():
            start_match = start_pattern.search(line)
            if start_match:
                current_start = float(start_match.group(1))
                continue

            end_match = end_pattern.search(line)
            if end_match and current_start is not None:
                silence_end = float(end_match.group(1))
                if silence_end > current_start:
                    silence_ranges.append((current_start, silence_end))
                current_start = None

        if current_start is not None and duration > current_start:
            silence_ranges.append((current_start, duration))

        return silence_ranges

    def _calculate_smart_split_points(self, duration: float,
                                      silence_ranges: List[Tuple[float, float]]) -> List[float]:
        """Pick split points near silence while preserving practical segment sizes."""
        split_points = []
        previous_point = 0.0
        ideal_point = float(self.split_duration_sec)
        min_segment_duration = min(
            max(30.0, self.split_duration_sec * 0.2),
            max(1.0, self.split_duration_sec * 0.8)
        )
        search_window = min(300.0, max(10.0, self.split_duration_sec * 0.15))
        max_silence_overrun = min(5.0, max(2.5, self.split_duration_sec * 0.02))

        while ideal_point < duration:
            if duration - ideal_point < min_segment_duration:
                merged_tail_duration = duration - previous_point
                if merged_tail_duration <= self.split_duration_sec + max_silence_overrun:
                    break

            best_point = None
            best_distance = float("inf")
            latest_allowed_point = previous_point + self.split_duration_sec + max_silence_overrun

            for silence_start, silence_end in silence_ranges:
                midpoint = (silence_start + silence_end) / 2
                if midpoint <= previous_point + min_segment_duration:
                    continue
                if midpoint > latest_allowed_point:
                    continue
                if duration - midpoint < min_segment_duration:
                    continue

                distance = abs(midpoint - ideal_point)
                if distance <= search_window and distance < best_distance:
                    best_distance = distance
                    best_point = midpoint

            split_point = best_point if best_point is not None else ideal_point
            split_point = round(split_point, 3)
            if duration - split_point < min_segment_duration:
                merged_tail_duration = duration - previous_point
                if merged_tail_duration <= self.split_duration_sec + max_silence_overrun:
                    break
            if split_point <= previous_point + 0.001:
                break

            split_points.append(split_point)
            previous_point = split_point
            ideal_point = split_point + self.split_duration_sec

        return split_points

    def _build_segment_ranges(self, duration: float,
                              split_points: List[float]) -> List[Tuple[float, float]]:
        points = [0.0]
        for point in split_points:
            if 0 < point < duration and point > points[-1] + 0.001:
                points.append(point)
        points.append(duration)

        ranges = []
        for index in range(len(points) - 1):
            start = round(points[index], 3)
            end = round(points[index + 1], 3)
            if end > start + 0.001:
                ranges.append((start, end))
        return ranges

    def _chunk_extension_for_audio(self, audio_path: str) -> str:
        extension = os.path.splitext(audio_path)[1].lower()
        return extension or ".mka"

    def _export_audio_segment(self, audio_path: str, chunk_path: str,
                              start: float, end: float):
        duration = max(0.001, end - start)
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-i", audio_path,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-map", "0:a:0",
            "-vn",
            "-c:a", "copy",
            "-avoid_negative_ts", "make_zero",
            chunk_path
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            startupinfo=self._startupinfo()
        )

    def _startupinfo(self):
        if sys.platform != "win32":
            return None
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return startupinfo

    def _ensure_chunk_offsets(self):
        if len(self.chunk_offsets) == len(self.temp_chunks):
            try:
                self.chunk_offsets = [float(offset) for offset in self.chunk_offsets]
                return
            except (TypeError, ValueError):
                pass

        self.chunk_offsets = [
            round(index * self.split_duration_sec, 3)
            for index in range(len(self.temp_chunks))
        ]

    def _get_chunk_offset(self, chunk_index: int) -> float:
        self._ensure_chunk_offsets()
        if 0 <= chunk_index < len(self.chunk_offsets):
            return float(self.chunk_offsets[chunk_index])
        return round(chunk_index * self.split_duration_sec, 3)

    def _copy_transcript_with_offset(self, transcript_json: dict, offset: float) -> dict:
        adjusted = transcript_json.copy()
        if "words" not in transcript_json:
            return adjusted

        adjusted_words = []
        for word in transcript_json.get("words", []):
            adjusted_word = word.copy()
            for key in ("start", "end"):
                if isinstance(adjusted_word.get(key), (int, float)):
                    adjusted_word[key] = round(adjusted_word[key] + offset, 3)
            adjusted_words.append(adjusted_word)

        adjusted["words"] = adjusted_words
        return adjusted

    def _cleanup_owned_chunk_artifacts(self):
        owned_paths = list(getattr(self, "owned_temp_chunks", []))
        for chunk_path in owned_paths:
            try:
                if chunk_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)
                if chunk_path:
                    json_path = os.path.splitext(chunk_path)[0] + ".json"
                    if os.path.exists(json_path):
                        os.remove(json_path)
            except OSError:
                pass

        self.owned_temp_chunks = []
        self._remove_temp_chunk_dir(log=False, owned_paths=owned_paths)

    def _remove_temp_chunk_dir(self, log: bool, owned_paths: Optional[List[str]] = None):
        temp_chunk_dir = getattr(self, "temp_chunk_dir", None)
        if not temp_chunk_dir:
            return

        if not self._is_owned_temp_chunk_dir(temp_chunk_dir, owned_paths):
            if log:
                self.log_message.emit(f"跳过未知临时切片目录: {temp_chunk_dir}")
            self.temp_chunk_dir = None
            return

        if os.path.isdir(temp_chunk_dir):
            try:
                shutil.rmtree(temp_chunk_dir)
                if log:
                    self.log_message.emit(f"已删除临时切片目录: {os.path.basename(temp_chunk_dir)}")
            except OSError as e:
                if log:
                    self.log_message.emit(f"清理临时切片目录失败: {e}")
        self.temp_chunk_dir = None

    def _is_owned_temp_chunk_dir(self, path: str, owned_paths: Optional[List[str]] = None) -> bool:
        try:
            resolved_dir = os.path.abspath(path)
            name = os.path.basename(os.path.normpath(resolved_dir))
            if "_chunks_" not in name:
                return False

            paths = owned_paths if owned_paths is not None else getattr(self, "owned_temp_chunks", [])
            paths = [chunk_path for chunk_path in paths if chunk_path]
            if paths:
                prefix = os.path.normcase(resolved_dir.rstrip(os.sep) + os.sep)
                return all(
                    os.path.normcase(os.path.abspath(chunk_path)).startswith(prefix)
                    for chunk_path in paths
                )

            allowed_parents = {
                os.path.normcase(os.path.dirname(os.path.abspath(path_value)))
                for path_value in (getattr(self, "file_path", None), getattr(self, "original_file_path", None))
                if path_value
            }
            return os.path.normcase(os.path.dirname(resolved_dir)) in allowed_parents
        except (TypeError, ValueError, OSError):
            return False

    def _process_next_chunk(self):
        """处理下一个待处理的音频片段。"""
        if self._is_cancelled:
            self.error.emit("任务被用户取消。")
            self._cleanup_chunks(force_cleanup=True)  # 用户取消时强制清理
            return

        # 检查是否使用异步处理
        if self.enable_async_processing and self.total_chunks > 1:
            self._process_chunks_async()
        else:
            # 使用原有的顺序处理逻辑（兼容性保证）
            self._process_chunks_sequential()

    def _process_chunks_async(self):
        """异步处理所有音频片段"""
        self.log_message.emit("-" * 20)
        self.log_message.emit(f"启用异步处理模式，并发处理 {self.total_chunks} 个片段...")
        self.chunk_progress.emit(-1, "async_start", f"异步处理 {self.total_chunks} 个片段")
        self.async_base_chunk_index = 0

        # 创建异步处理器
        self.async_processor = AsyncChunkProcessor(
            max_concurrent_chunks=self.max_concurrent_chunks,
            max_retries=self.max_retries
        )
        # 设置API速率限制
        self.async_processor.max_requests_per_minute = self.api_rate_limit_per_minute

        # 连接信号
        self.async_processor.chunk_started.connect(self._on_async_chunk_started)
        self.async_processor.chunk_completed.connect(self._on_async_chunk_completed)
        self.async_processor.chunk_failed.connect(self._on_async_chunk_failed)
        self.async_processor.all_chunks_completed.connect(self._on_async_all_completed)
        self.async_processor.processing_failed.connect(self._on_async_processing_failed)
        self.async_processor.progress_updated.connect(self._on_async_progress_updated)

        # 启动异步处理
        success = self.async_processor.process_chunks_async(
            chunk_paths=self.temp_chunks,
            split_duration_sec=self.split_duration_sec,
            language_code=self.language_code,
            tag_audio_events=self.tag_audio_events,
            ffmpeg_available=self.ffmpeg_available,
            log_callback=lambda msg: self.log_message.emit(msg),
            chunk_indices=list(range(self.total_chunks)),
            chunk_offsets=self.chunk_offsets
        )

        if not success:
            self.error.emit("启动异步处理失败")

    def _process_chunks_sequential(self):
        """顺序处理音频片段（原有逻辑）"""
        if self.current_chunk_index < self.total_chunks:
            self.time_offset = self._get_chunk_offset(self.current_chunk_index)
            chunk_path = self.temp_chunks[self.current_chunk_index]

            self.log_message.emit("-" * 20)
            self.log_message.emit(f"正在处理片段 {self.current_chunk_index + 1}/{self.total_chunks}: {os.path.basename(chunk_path)}")
            self.chunk_progress.emit(self.current_chunk_index, "processing", f"正在处理片段 {self.current_chunk_index + 1}/{self.total_chunks}")

            self._process_single_file(chunk_path)
        else:
            self._finalize_task()

    def _process_restored_chunks(self):
        """处理恢复的任务"""
        # 计算剩余需要处理的片段
        remaining_chunks = self.temp_chunks[self.current_chunk_index:]

        if not remaining_chunks:
            # 所有片段都已完成，直接进入最终处理
            self.log_message.emit("所有片段已在之前完成，直接生成最终文件...")
            self._finalize_task()
            return

        self.log_message.emit(f"需要继续处理 {len(remaining_chunks)} 个剩余片段...")

        # 根据剩余片段数量选择处理模式
        if self.enable_async_processing and len(remaining_chunks) > 1:
            # 异步处理剩余片段
            self._process_remaining_chunks_async(remaining_chunks)
        else:
            # 顺序处理剩余片段
            self._process_chunks_sequential()

    def _process_remaining_chunks_async(self, remaining_chunks: List[str]):
        """异步处理剩余的音频片段"""
        self.log_message.emit("-" * 20)
        self.log_message.emit(f"恢复模式：异步处理剩余 {len(remaining_chunks)} 个片段...")
        self.chunk_progress.emit(-1, "async_restore", f"恢复异步处理 {len(remaining_chunks)} 个片段")
        self.async_base_chunk_index = self.current_chunk_index
        remaining_indices = list(range(self.current_chunk_index, self.total_chunks))

        # 创建异步处理器
        self.async_processor = AsyncChunkProcessor(
            max_concurrent_chunks=self.max_concurrent_chunks,
            max_retries=self.max_retries
        )
        # 设置API速率限制
        self.async_processor.max_requests_per_minute = self.api_rate_limit_per_minute

        # 连接信号
        self.async_processor.chunk_started.connect(self._on_async_chunk_started_restored)
        self.async_processor.chunk_completed.connect(self._on_async_chunk_completed_restored)
        self.async_processor.chunk_failed.connect(self._on_async_chunk_failed)
        self.async_processor.all_chunks_completed.connect(self._on_async_all_completed_restored)
        self.async_processor.processing_failed.connect(self._on_async_processing_failed)
        self.async_processor.progress_updated.connect(self._on_async_progress_updated_restored)

        # 启动异步处理剩余片段
        success = self.async_processor.process_chunks_async(
            chunk_paths=remaining_chunks,
            split_duration_sec=self.split_duration_sec,
            language_code=self.language_code,
            tag_audio_events=self.tag_audio_events,
            ffmpeg_available=self.ffmpeg_available,
            log_callback=lambda msg: self.log_message.emit(msg),
            chunk_indices=remaining_indices,
            chunk_offsets=self.chunk_offsets
        )

        if not success:
            self.error.emit("恢复模式：启动异步处理失败")

    def _process_single_file(self, file_path: str):
        """为单个文件准备并开始上传任务。"""
        self.uploader = self.client.prepare_upload_task(
            file_path, self.language_code, self.tag_audio_events,
            max_retries=self.max_retries
        )
        if not self.uploader:
            self.error.emit(f"为文件 {os.path.basename(file_path)} 准备任务失败。")
            return

        self.uploader.signals.progress.connect(self.progress_updated)
        self.uploader.signals.finished.connect(self.on_upload_finished)
        self.uploader.signals.error.connect(self.on_chunk_error)
        
        QThreadPool.globalInstance().start(self.uploader)

    # === 异步处理信号回调方法 ===
    def _on_async_chunk_started(self, chunk_index: int):
        """异步片段开始处理回调"""
        self.log_message.emit(f"开始异步处理片段 {chunk_index + 1}/{self.total_chunks}")
        self.chunk_progress.emit(chunk_index, "started", f"开始处理片段 {chunk_index + 1}/{self.total_chunks}")

    def _on_async_chunk_completed(self, chunk_index: int, transcript_json: dict):
        """异步片段完成回调"""
        self.log_message.emit(f"片段 {chunk_index + 1}/{self.total_chunks} 异步转录成功")
        self.chunk_progress.emit(chunk_index, "completed", f"片段 {chunk_index + 1}/{self.total_chunks} 转录完成")

    def _on_async_chunk_failed(self, chunk_index: int, error_message: str):
        """异步片段失败回调"""
        self.log_message.emit(f"片段 {chunk_index + 1}/{self.total_chunks} 处理失败: {error_message}")
        self.chunk_progress.emit(chunk_index, "failed", f"片段 {chunk_index + 1}/{self.total_chunks} 处理失败")

    def _on_async_all_completed(self, combined_transcript: dict):
        """所有异步片段完成回调"""
        self.log_message.emit("所有片段异步处理完成，正在合并结果...")
        self.combined_transcript = combined_transcript

        # 确保进度显示完成
        self.chunk_progress.emit(-1, "completed", "异步处理完成，正在生成字幕文件...")

        # 调用最终处理
        self._finalize_task()

    def _on_async_processing_failed(self, error_message: str):
        """异步处理失败回调"""
        self.log_message.emit(f"异步处理失败: {error_message}")

        if self._is_cancelled:
            if not self._cancellation_reported:
                self._cancellation_reported = True
                self.error.emit("用户取消了任务")
            return

        # 检查是否可以降级到顺序处理
        if self.async_processor:
            progress_info = self.async_processor.get_progress_info()
            if progress_info.get("is_cancelled"):
                if not self._cancellation_reported:
                    self._cancellation_reported = True
                    self.error.emit("用户取消了任务")
                return

            completed_count = progress_info.get("completed_chunks", 0)

            if completed_count > 0:
                # 有部分片段成功，尝试降级处理
                self.log_message.emit(f"已完成 {completed_count} 个片段，尝试降级到顺序处理剩余片段...")
                self._fallback_to_sequential_processing()
            else:
                # 完全失败，报告错误
                self.error.emit(f"异步处理完全失败: {error_message}")
        else:
            self.error.emit(f"异步处理失败: {error_message}")

    def _fallback_to_sequential_processing(self):
        """降级到顺序处理模式"""
        try:
            if self._is_cancelled:
                return

            self.log_message.emit("正在降级到顺序处理模式...")

            # 禁用异步处理
            self.enable_async_processing = False

            # 获取已完成的片段信息
            if self.async_processor:
                completed_chunks = self.async_processor.completed_chunks
                base_index = self.async_base_chunk_index

                # 只复用从 base_index 开始的连续成功前缀。
                # 如果中间片段失败，后面的成功片段会在顺序模式中重跑，避免字幕缺段。
                reusable_indices = []
                next_index = base_index
                while next_index < self.total_chunks and next_index in completed_chunks:
                    reusable_indices.append(next_index)
                    next_index += 1

                if reusable_indices:
                    self.log_message.emit(
                        f"复用连续完成的 {len(reusable_indices)} 个片段，"
                        f"将从第 {next_index + 1} 个片段顺序补处理..."
                    )
                    for chunk_index in reusable_indices:
                        self._append_transcript(completed_chunks[chunk_index])
                else:
                    self.log_message.emit(
                        f"没有可安全复用的连续片段，将从第 {base_index + 1} 个片段顺序补处理..."
                    )

                self.current_chunk_index = next_index

                # 清理异步处理器
                self.async_processor = None

            # 继续顺序处理剩余片段
            if self.current_chunk_index < self.total_chunks:
                self.log_message.emit(f"继续顺序处理剩余 {self.total_chunks - self.current_chunk_index} 个片段...")
                self._process_chunks_sequential()
            else:
                # 所有片段都已完成
                self._finalize_task()

        except Exception as e:
            self.error.emit(f"降级处理失败: {e}")

    def _append_transcript(self, transcript_json: dict):
        """按顺序追加一个已经带全局时间偏移的转录结果。"""
        if not transcript_json:
            return

        if not self.combined_transcript:
            self.combined_transcript = transcript_json.copy()
            if "words" in self.combined_transcript:
                self.combined_transcript["words"] = list(self.combined_transcript.get("words", []))
            return

        words = transcript_json.get("words", [])
        self.combined_transcript.setdefault("words", [])
        self.combined_transcript["words"].extend(words)

        text = transcript_json.get("text", "")
        self.combined_transcript.setdefault("text", "")
        if text:
            if self.combined_transcript["text"]:
                self.combined_transcript["text"] += " "
            self.combined_transcript["text"] += text

    def _on_async_progress_updated(self, chunk_index: int, bytes_sent: int, total_bytes: int):
        """异步处理进度更新回调"""
        # 转发进度信号到主窗口
        self.progress_updated.emit(bytes_sent, total_bytes)

        # 更新当前处理的片段索引（用于UI显示）
        self.current_chunk_index = chunk_index

    # === 恢复模式的异步处理回调方法 ===
    def _on_async_chunk_started_restored(self, chunk_index: int):
        """恢复模式：异步片段开始处理回调"""
        self.log_message.emit(f"恢复模式：开始异步处理片段 {chunk_index + 1}/{self.total_chunks}")
        self.chunk_progress.emit(chunk_index, "started", f"恢复处理片段 {chunk_index + 1}/{self.total_chunks}")

    def _on_async_chunk_completed_restored(self, chunk_index: int, transcript_json: dict):
        """恢复模式：异步片段完成回调"""
        self.log_message.emit(f"恢复模式：片段 {chunk_index + 1}/{self.total_chunks} 异步转录成功")
        self.chunk_progress.emit(chunk_index, "completed", f"恢复片段 {chunk_index + 1}/{self.total_chunks} 转录完成")
        # transcript_json 在这里不需要特殊处理，由异步处理器自动合并

    def _on_async_all_completed_restored(self, remaining_transcript: dict):
        """恢复模式：所有剩余异步片段完成回调"""
        self.log_message.emit("恢复模式：所有剩余片段异步处理完成，正在合并结果...")

        # 合并已有的转录结果和新的转录结果
        # 注意：remaining_transcript中的时间偏移已经在异步处理器中正确处理
        if self.combined_transcript and self.combined_transcript.get("words"):
            # 已有部分转录结果，需要合并
            self.combined_transcript["words"].extend(remaining_transcript.get("words", []))
            if self.combined_transcript["text"]:
                self.combined_transcript["text"] += " "
            self.combined_transcript["text"] += remaining_transcript.get("text", "")
        else:
            # 没有已有结果，直接使用新结果
            self.combined_transcript = remaining_transcript

        self._finalize_task()

    def _on_async_progress_updated_restored(self, chunk_index: int, bytes_sent: int, total_bytes: int):
        """恢复模式：异步处理进度更新回调"""
        # 转发进度信号到主窗口
        self.progress_updated.emit(bytes_sent, total_bytes)
        # chunk_index 用于标识片段，但在恢复模式下不需要特殊处理

    def on_upload_finished(self, transcript_json):
        """当一个片段上传和转录成功时调用。"""
        self.log_message.emit(f"片段 {self.current_chunk_index + 1}/{self.total_chunks} 转录成功。")
        adjusted_transcript = self._copy_transcript_with_offset(
            transcript_json,
            self._get_chunk_offset(self.current_chunk_index)
        )
        
        # 保存分段的 JSON 文件以供调试
        try:
            chunk_path = self.temp_chunks[self.current_chunk_index]
            base_chunk_path, _ = os.path.splitext(chunk_path)
            segment_json_path = base_chunk_path + ".json"
            with open(segment_json_path, 'w', encoding='utf-8') as f:
                json.dump(adjusted_transcript, f, ensure_ascii=False, indent=4)
            self.log_message.emit(f"分段转录JSON已保存到: {os.path.basename(segment_json_path)}")
        except Exception as e:
            self.log_message.emit(f"警告：保存分段JSON文件失败: {e}")

        self._append_transcript(adjusted_transcript)
        
        self.current_chunk_index += 1
        
        if self.current_chunk_index < self.total_chunks:
            self._process_next_chunk()
        else:
            # 这是最后一个片段，直接进入最终处理阶段
            self._finalize_task()

    def on_chunk_error(self, error_message: str):
        """当处理片段出错时调用。"""
        self.error.emit(f"处理片段 {self.current_chunk_index + 1}/{self.total_chunks} 时出错: {error_message}")

    def _finalize_task(self):
        """所有片段处理完毕后，合并结果并生成最终文件。"""
        self.log_message.emit("-" * 20)
        self.log_message.emit("所有片段处理完毕，正在生成最终文件...")
        
        base_path, _ = os.path.splitext(self.original_file_path)
        output_json_path = base_path + ".json"
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.combined_transcript, f, ensure_ascii=False, indent=4)
            self.log_message.emit(f"合并后的转录文本已保存到:\n{output_json_path}")
        except Exception as e:
            self.error.emit(f"保存合并后的 JSON 文件时出错: {e}")
            # 保存失败时不清理临时文件，以便重试
            return

        self.log_message.emit("正在生成SRT字幕文件...")
        srt_data = create_srt_from_json(
            self.combined_transcript,
            max_subtitle_duration=self.max_subtitle_duration,
            subtitle_settings=self.subtitle_settings
        )
        if not srt_data:
            self.error.emit("从合并后的JSON生成SRT失败。")
            # SRT生成失败时不清理临时文件，以便重试
            return

        output_srt_path = base_path + ".srt"
        task_success = False
        try:
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_data)
            self.log_message.emit(f"最终SRT字幕文件已保存到:\n{output_srt_path}")

            # 在单文件处理模式下，清理冗余的临时JSON文件
            self._cleanup_temporary_json_files()

            task_success = True
            self.finished.emit("任务成功完成！")
        except Exception as e:
            self.error.emit(f"保存最终SRT文件时出错: {e}")
            # 保存失败时不清理临时文件，以便重试
        finally:
            # 只有在任务成功完成时才清理临时文件
            if task_success:
                self._cleanup_chunks(force_cleanup=True)

    def _cleanup_temporary_json_files(self):
        """清理单文件处理模式下的冗余临时JSON文件"""
        if self.total_chunks == 1:
            # 单文件处理模式：检查是否需要清理临时JSON文件
            try:
                chunk_path = self.temp_chunks[0]
                base_chunk_path, _ = os.path.splitext(chunk_path)
                temp_json_path = base_chunk_path + ".json"

                # 计算最终JSON文件路径
                base_path, _ = os.path.splitext(self.original_file_path)
                final_json_path = base_path + ".json"

                # 只有当临时JSON文件路径与最终JSON文件路径不同时才删除
                if (os.path.exists(temp_json_path) and
                    os.path.abspath(temp_json_path) != os.path.abspath(final_json_path)):
                    os.remove(temp_json_path)
                    self.log_message.emit(f"已清理临时JSON文件: {os.path.basename(temp_json_path)}")
                elif os.path.abspath(temp_json_path) == os.path.abspath(final_json_path):
                    self.log_message.emit("单文件模式：临时JSON文件即为最终文件，无需清理")

            except (OSError, IndexError) as e:
                self.log_message.emit(f"清理临时JSON文件时出错: {e}")
        else:
            # 多片段处理模式：保留临时JSON文件用于调试
            self.log_message.emit("多片段处理模式：保留临时JSON文件用于调试")

    def request_cancellation(self):
        """请求取消当前任务。"""
        self.log_message.emit("正在取消上传...")
        self._is_cancelled = True

        # 取消异步处理器
        if self.async_processor:
            self.async_processor.cancel()

        # 取消当前上传器
        if self.uploader:
            self.uploader.cancel()

        # 用户取消时强制清理临时文件
        self._cleanup_chunks(force_cleanup=True)

    def _cleanup_chunks(self, force_cleanup=False):
        """清理所有临时的音频片段文件。

        Args:
            force_cleanup: 强制清理，即使任务可能需要重试
        """
        # 如果不是强制清理且任务可能需要重试，则跳过清理
        if not force_cleanup and not self._is_cancelled:
            self.log_message.emit("任务可能需要重试，保留临时文件...")
            return

        self.log_message.emit("正在清理所有临时音频片段...")

        # 清理音频片段文件
        owned_chunk_paths = list(self.owned_temp_chunks)
        if self.owned_temp_chunks:
            for chunk_path in self.owned_temp_chunks:
                try:
                    if chunk_path and os.path.exists(chunk_path):
                        os.remove(chunk_path)
                        self.log_message.emit(f"已删除临时片段: {os.path.basename(chunk_path)}")

                    # 同时删除对应的JSON文件
                    if chunk_path:
                        json_path = os.path.splitext(chunk_path)[0] + ".json"
                        if os.path.exists(json_path):
                            os.remove(json_path)
                            self.log_message.emit(f"已删除片段JSON: {os.path.basename(json_path)}")

                except (OSError, TypeError) as e:
                    chunk_name = os.path.basename(chunk_path) if chunk_path else "未知文件"
                    self.log_message.emit(f"清理文件 {chunk_name} 失败: {e}")
            self.owned_temp_chunks = []

        self._remove_temp_chunk_dir(log=True, owned_paths=owned_chunk_paths)

        # 清理提取的音频文件（如果是从视频提取的）
        if (hasattr(self, 'original_file_path') and self.original_file_path and
            hasattr(self, 'file_path') and self.file_path and
            self.original_file_path != self.file_path):
            try:
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                    self.log_message.emit(f"已删除提取的音频文件: {os.path.basename(self.file_path)}")
            except (OSError, TypeError) as e:
                self.log_message.emit(f"清理提取的音频文件失败: {e}")

        self.log_message.emit("临时文件清理完成。")
