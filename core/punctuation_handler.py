#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一标点符号处理模块
为整个系统提供一致的标点符号识别和处理逻辑
"""

from typing import Set, List, Tuple


class PunctuationHandler:
    """
    统一标点符号处理器
    
    提供系统级的标点符号识别、分类和处理功能
    解决韩语标点符号处理问题，同时保持对所有语言的兼容性
    """
    
    # 高优先级标点符号（句子结束符）
    HIGH_PRIORITY_PUNCTUATION = {
        # 拉丁语系句子结束符
        ".", "!", "?",
        # 东亚语系句子结束符  
        "。", "！", "？"
    }
    
    # 中优先级标点符号（子句结束符）
    MEDIUM_PRIORITY_PUNCTUATION = {
        # 拉丁语系子句结束符
        ";", ":", ")", "]", "}",
        # 东亚语系子句结束符，包含引用结束符
        "；", "：", "》", "」", "】", "）"
    }
    
    # 低优先级标点符号（短语分隔符）
    LOW_PRIORITY_PUNCTUATION = {
        # 拉丁语系短语分隔符
        ",", "(", "[", "{", "...", "…", "-",
        # 东亚语系短语分隔符，包含引用开始符、省略号和连字符
        "，", "、", "《", "「", "【", "（"
    }

    # 句尾标点后常见的闭合符号。识别分割点时应透过这些符号检查前一个字符。
    TRAILING_CLOSERS = {
        '"', "'", "”", "’", "」", "』", "》", "）", ")", "]", "}", "】"
    }
    
    # 所有标点符号的并集
    ALL_PUNCTUATION = (
        HIGH_PRIORITY_PUNCTUATION |
        MEDIUM_PRIORITY_PUNCTUATION |
        LOW_PRIORITY_PUNCTUATION |
        TRAILING_CLOSERS
    )
    
    # 用于文本分割的字符（包含空格）
    SPLIT_CHARACTERS = " " + "".join(ALL_PUNCTUATION)
    
    @classmethod
    def get_punctuation_priority(cls, punct: str) -> int:
        """
        获取标点符号的优先级
        
        Args:
            punct: 标点符号
            
        Returns:
            优先级 (0=高, 1=中, 2=低, -1=不是分割标点)
        """
        if punct in cls.HIGH_PRIORITY_PUNCTUATION:
            return 0
        elif punct in cls.MEDIUM_PRIORITY_PUNCTUATION:
            return 1
        elif punct in cls.LOW_PRIORITY_PUNCTUATION:
            return 2
        else:
            return -1
    
    @classmethod
    def is_punctuation(cls, char: str) -> bool:
        """
        检查字符是否为标点符号
        
        Args:
            char: 要检查的字符
            
        Returns:
            是否为标点符号
        """
        return char in cls.ALL_PUNCTUATION
    
    @classmethod
    def get_split_characters(cls) -> str:
        """
        获取用于文本分割的字符集
        
        Returns:
            包含空格和所有标点符号的字符串
        """
        return cls.SPLIT_CHARACTERS
    
    @classmethod
    def get_high_priority_punctuation(cls) -> Set[str]:
        """获取高优先级标点符号集合"""
        return cls.HIGH_PRIORITY_PUNCTUATION.copy()
    
    @classmethod
    def get_medium_priority_punctuation(cls) -> Set[str]:
        """获取中优先级标点符号集合"""
        return cls.MEDIUM_PRIORITY_PUNCTUATION.copy()
    
    @classmethod
    def get_low_priority_punctuation(cls) -> Set[str]:
        """获取低优先级标点符号集合"""
        return cls.LOW_PRIORITY_PUNCTUATION.copy()
    
    @classmethod
    def get_all_punctuation(cls) -> Set[str]:
        """获取所有标点符号集合"""
        return cls.ALL_PUNCTUATION.copy()
    
    @classmethod
    def word_ends_with_punctuation(cls, text: str) -> Tuple[bool, str, int]:
        """
        检查文本是否以标点符号结尾
        
        Args:
            text: 要检查的文本
            
        Returns:
            (是否以标点结尾, 标点符号, 优先级)
        """
        text = text.strip()
        if not text:
            return False, "", -1
        
        check_index = len(text) - 1
        while check_index > 0 and text[check_index] in cls.TRAILING_CLOSERS:
            check_index -= 1

        last_char = text[check_index]
        priority = cls.get_punctuation_priority(last_char)
        
        if priority >= 0:
            return True, last_char, priority
        
        return False, "", -1
    
    @classmethod
    def find_split_position(cls, text: str, max_length: int) -> int:
        """
        在文本中寻找最佳分割位置
        
        Args:
            text: 要分割的文本
            max_length: 最大长度
            
        Returns:
            最佳分割位置
        """
        if len(text) <= max_length:
            return len(text)
        
        split_chars = cls.get_split_characters()
        
        # 从最大长度向前搜索分割点
        best_pos = -1
        search_end = min(max_length + 1, len(text))
        
        for i in range(search_end - 1, 0, -1):
            if text[i] in split_chars:
                # 对于空格，在空格前分割
                if text[i] == ' ':
                    best_pos = i
                    break
                # 对于标点符号，在标点符号后分割
                else:
                    if i + 1 <= max_length:
                        best_pos = i + 1
                        break
        
        # 如果没找到合适的分割点，强制在最大长度处分割
        if best_pos <= 0:
            best_pos = max_length
        
        return best_pos
    
    @classmethod
    def is_sentence_ending(cls, text: str) -> bool:
        """
        检查文本是否以句子结束符结尾
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否以句子结束符结尾
        """
        has_punct, punct, priority = cls.word_ends_with_punctuation(text)
        return has_punct and priority == 0  # 只有高优先级标点符号才算句子结束符
