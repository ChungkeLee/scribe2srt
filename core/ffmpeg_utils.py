# -*- coding: utf-8 -*-

"""
这个文件包含了所有与 FFmpeg 和 ffprobe 交互的工具函数。
"""

import sys
import subprocess
import shutil
import json
import os
import math
from typing import Optional, Dict, Any


def _parse_optional_float(value: Any) -> Optional[float]:
    """将 ffprobe 的数值字段安全转换为 float。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and math.isfinite(parsed) else None


def _parse_optional_int(value: Any) -> Optional[int]:
    """将 ffprobe 的整数或数字字符串字段安全转换为 int。"""
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _probe_raw_aac_duration(media_file_path: str, startupinfo=None,
                            cancel_check=None) -> Optional[float]:
    """Count ADTS AAC frames when container duration is only an estimate."""
    try:
        process = subprocess.Popen(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "a:0",
                "-show_entries", "stream=nb_read_frames,sample_rate",
                "-of", "json", media_file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            startupinfo=startupinfo,
        )
        termination_requested = False
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_check and cancel_check() and process.poll() is None:
                    if termination_requested:
                        process.kill()
                    else:
                        process.terminate()
                        termination_requested = True
        if cancel_check and cancel_check():
            return None
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=stdout,
                stderr=stderr,
            )
        stream = json.loads(stdout).get("streams", [{}])[0]
        frame_count = _parse_optional_int(stream.get("nb_read_frames"))
        sample_rate = _parse_optional_int(stream.get("sample_rate"))
        if frame_count and sample_rate:
            # AAC-LC/ADTS uses 1024 PCM samples per coded audio frame.  Counting
            # frames is substantially more reliable than the bitrate-based format
            # duration estimate used by ffprobe for raw ADTS streams.
            return frame_count * 1024 / sample_rate
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, OSError):
        return None
    return None

def is_ffmpeg_available() -> bool:
    """检查 FFmpeg 是否在系统的 PATH 中可用。"""
    return shutil.which("ffmpeg") is not None

def get_media_info(media_file_path: str, log_callback=None,
                   cancel_check=None) -> Optional[Dict[str, Any]]:
    """使用 ffprobe 获取切片决策所需的音频流和容器信息。

    保留原有 ``duration``、``codec`` 返回字段，同时补充容器、码率、
    采样率和声道数，供调用方按真实编码而不是文件扩展名选择切片策略。
    """
    if not is_ffmpeg_available():
        if log_callback:
            log_callback("  FFmpeg/ffprobe 不可用，跳过媒体信息检测。")
        return None
    
    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries",
            "stream=codec_name,duration,bit_rate,sample_rate,channels:"
            "format=format_name,duration,bit_rate",
            "-of", "json",
            media_file_path
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            startupinfo=startupinfo
        )
        
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        format_info = data.get("format", {})

        duration = _parse_optional_float(format_info.get("duration"))
        if duration is None:
            duration = _parse_optional_float(stream.get("duration"))

        stream_bit_rate = _parse_optional_int(stream.get("bit_rate"))
        format_bit_rate = _parse_optional_int(format_info.get("bit_rate"))
        codec_name = str(stream.get("codec_name") or "N/A").lower()
        format_name = str(format_info.get("format_name") or "").lower()

        if codec_name == "aac" and format_name == "aac":
            counted_duration = _probe_raw_aac_duration(
                media_file_path,
                startupinfo=startupinfo,
                cancel_check=cancel_check,
            )
            if counted_duration is not None:
                duration = counted_duration

        return {
            "duration": duration or 0.0,
            "codec": codec_name,
            "format": format_name,
            # ``container`` 是语义更清晰的别名；``format`` 保留给简洁调用。
            "container": format_name,
            "bit_rate": stream_bit_rate or format_bit_rate,
            "sample_rate": _parse_optional_int(stream.get("sample_rate")),
            "channels": _parse_optional_int(stream.get("channels")),
        }

    except FileNotFoundError:
        if log_callback:
            log_callback("  错误: ffprobe 未找到。")
        return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        if log_callback:
            log_callback(f"  使用 ffprobe 获取信息失败: {e}")
        return None

def extract_audio(video_path: str, output_path: str, log_callback=None) -> bool:
    """使用 FFmpeg 从视频文件中无损提取音频流。"""
    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        command = ["ffmpeg", "-i", video_path, "-vn", "-c:a", "copy", "-y", output_path]
        
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', startupinfo=startupinfo)
        
        if log_callback:
            log_callback(f"音频提取成功: {os.path.basename(output_path)}")
        return True
    except FileNotFoundError:
        if log_callback:
            log_callback("FFmpeg 未找到。请确保它已安装并位于系统的PATH中。")
        return False
    except subprocess.CalledProcessError as e:
        error_message = "FFmpeg 提取音频失败。\n"
        error_message += "这可能是因为视频文件已损坏、不包含音频流或编码与容器不兼容。\n"
        error_message += f"返回码: {e.returncode}\n"
        try:
            stderr_output = e.stderr.strip()
            error_message += f"FFmpeg 输出:\n{stderr_output}"
        except Exception as decode_error:
            error_message += f"(无法解码 FFmpeg 的错误输出: {decode_error})"
        if log_callback:
            log_callback(error_message)
        return False
    except Exception as e:
        if log_callback:
            log_callback(f"提取音频时发生未知错误: {e}")
        return False
