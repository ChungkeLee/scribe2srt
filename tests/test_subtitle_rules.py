#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
字幕规则快速测试脚本
专门用于测试字幕是否符合各项专业标准
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

try:
    from .optimize_based_on_analysis import EnhancedSubtitleAnalyzer
except ImportError:
    from optimize_based_on_analysis import EnhancedSubtitleAnalyzer

def analyze_single_file(srt_path: str):
    """测试单个SRT文件"""
    if not os.path.exists(srt_path):
        print(f"文件不存在: {srt_path}")
        return
    
    analyzer = EnhancedSubtitleAnalyzer()
    result = analyzer.analyze_subtitle_rules(srt_path)
    
    if 'error' in result:
        print(f"分析出错: {result['error']}")
        return
    
    print("=" * 60)
    print(f"字幕规则测试报告: {result['file']}")
    print("=" * 60)
    
    stats = result['stats']
    violations = result['violations']
    rules = result['rules']
    
    # 基本统计
    print(f"字幕总数: {stats['total_subtitles']}")
    print(f"平均时长: {stats['avg_duration']:.2f}s")
    print(f"平均间隔: {stats['avg_gap']:.3f}s")
    print(f"平均CPS: {stats['avg_cps']:.1f}")
    
    # 语言分布
    lang_dist = stats['language_distribution']
    print(f"语言分布: CJK={lang_dist['cjk']}, Latin={lang_dist['latin']}")
    
    # 违规统计
    total_violations = sum(len(v) for v in violations.values())
    compliance_rate = ((stats['total_subtitles'] - total_violations) / stats['total_subtitles'] * 100) if stats['total_subtitles'] > 0 else 0
    
    print(f"\n合规率: {compliance_rate:.2f}%")
    print(f"违规总数: {total_violations}")
    
    if total_violations == 0:
        print("所有字幕都符合规则要求！")
        return
    
    print("\n违规详情:")
    print("-" * 40)
    
    # 详细违规信息
    violation_names = {
        'duration_too_short': '时长过短',
        'duration_too_long': '时长过长',
        'overlap': '时间重叠',
        'gap_too_small': '间隔过小',
        'cps_too_high': 'CPS过高',
        'cpl_exceeded': '行长超限',
        'too_many_lines': '行数过多',
        'punctuation_issues': '标点问题'
    }
    
    for violation_type, violation_list in violations.items():
        if violation_list:
            print(f"\n{violation_names.get(violation_type, violation_type)} ({len(violation_list)}个):")
            
            # 显示前5个示例
            for i, example in enumerate(violation_list[:5], 1):
                if violation_type == 'duration_too_short':
                    print(f"  {i}. #{example['number']}: {example['duration']:.3f}s < {rules['min_duration']:.3f}s")
                    print(f"     文本: {example['text']}")
                elif violation_type == 'duration_too_long':
                    print(f"  {i}. #{example['number']}: {example['duration']:.1f}s > {rules['max_duration']:.1f}s")
                    print(f"     文本: {example['text']}")
                elif violation_type == 'overlap':
                    print(f"  {i}. #{example['number']}-{example['next_number']}: 重叠 {-example['gap']:.3f}s")
                elif violation_type == 'gap_too_small':
                    print(f"  {i}. #{example['number']}-{example['next_number']}: {example['gap']:.3f}s < {rules['min_gap']:.3f}s")
                elif violation_type == 'cps_too_high':
                    print(f"  {i}. #{example['number']}: {example['cps']:.1f} > {example['max_cps']} ({example['language']})")
                    print(f"     文本: {example['text']}")
                elif violation_type == 'cpl_exceeded':
                    print(f"  {i}. #{example['number']}行{example['line_number']}: {example['length']} > {example['max_length']} ({example['language']})")
                    print(f"     行文本: {example['line_text']}")
                elif violation_type == 'too_many_lines':
                    print(f"  {i}. #{example['number']}: {example['lines']}行 > {rules['max_lines']}行")
                elif violation_type == 'punctuation_issues':
                    print(f"  {i}. #{example['number']}: 末尾'{example['last_char']}'")
                    print(f"     文本: {example['text'][:50]}{'...' if len(example['text']) > 50 else ''}")
            
            if len(violation_list) > 5:
                print(f"  ... 还有 {len(violation_list) - 5} 个类似问题")

def analyze_directory(directory: str):
    """测试目录中的所有SRT文件"""
    analyzer = EnhancedSubtitleAnalyzer()
    result = analyzer.analyze_directory_rules(directory)
    
    if 'error' in result:
        print(f"分析出错: {result['error']}")
        return
    
    analyzer.print_rules_analysis_report(result)
    if hasattr(analyzer, 'generate_improvement_suggestions'):
        analyzer.generate_improvement_suggestions(result)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='字幕规则测试工具')
    parser.add_argument('path', help='SRT文件路径或目录路径')
    parser.add_argument('--file', '-f', action='store_true', help='指定路径为文件（默认自动检测）')
    parser.add_argument('--dir', '-d', action='store_true', help='指定路径为目录（默认自动检测）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"路径不存在: {args.path}")
        return
    
    # 自动检测文件类型
    if args.file or (not args.dir and os.path.isfile(args.path)):
        analyze_single_file(args.path)
    elif args.dir or os.path.isdir(args.path):
        analyze_directory(args.path)
    else:
        print("无法确定路径类型，请使用 --file 或 --dir 参数")

if __name__ == "__main__":
    # 如果没有命令行参数，默认测试sample目录
    if len(sys.argv) == 1:
        print("使用默认参数测试 sample 目录...")
        analyze_directory("sample")
    else:
        main()
