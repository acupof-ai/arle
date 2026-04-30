#!/usr/bin/env python3
"""
ARLE轨迹质量评估与RL训练数据预处理
v2 schema轨迹分析器
"""

import json
import glob
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TrajectoryQuality:
    """轨迹质量评估结果"""
    task_completion: float  # 任务完成度 (0-1)
    tool_usage_efficiency: float  # 工具使用效率
    error_recovery: float  # 错误恢复能力
    response_relevance: float  # 响应相关性
    overall_score: float  # 总分

def analyze_trajectory(data: Dict[str, Any]) -> TrajectoryQuality:
    """分析单个轨迹的质量"""

    # 任务完成度
    if data['result']['terminal_state'] == 'stop':
        completion = 1.0
    elif data['result']['terminal_state'] == 'policy_short_circuit':
        completion = 0.5  # 部分完成
    else:
        completion = 0.0  # 未完成

    # 工具使用效率 (成功调用比例)
    tool_calls = sum(1 for m in data['messages']
                    if m['role'] == 'assistant' and
                    isinstance(m.get('content'), list) and
                    any(c.get('type') == 'tool_use' for c in m['content']))

    tool_errors = sum(1 for m in data['messages']
                     if m['role'] == 'tool' and '[stderr]' in m['content'])

    if tool_calls > 0:
        efficiency = max(0.0, 1.0 - (tool_errors / tool_calls))
    else:
        efficiency = 0.5 if completion > 0 else 0.0

    # 错误恢复能力
    repeated_errors = 0
    prev_error = None
    for m in data['messages']:
        if m['role'] == 'tool' and '[stderr]' in m['content']:
            if m['content'] == prev_error:
                repeated_errors += 1
            prev_error = m['content']

    recovery = max(0.0, 1.0 - (repeated_errors / max(1, tool_calls)))

    # 响应相关性 (简化评估)
    has_chinese_task = '中文' in data.get('user_input', '') or any(ord(c) > 127 for c in data.get('user_input', ''))
    has_tool_use_in_task = any(word in data.get('user_input', '').lower()
                              for word in ['shell', '命令', 'command', '文件', 'file'])

    if has_tool_use_in_task and tool_calls > 0:
        relevance = 1.0
    elif not has_tool_use_in_task and tool_calls == 0:
        relevance = 1.0
    else:
        relevance = 0.5

    # 计算总分
    overall = (completion * 0.4 + efficiency * 0.3 + recovery * 0.2 + relevance * 0.1)

    return TrajectoryQuality(completion, efficiency, recovery, relevance, overall)

def prepare_training_data(trajectory_file: str) -> Dict[str, Any]:
    """为RL训练准备轨迹数据"""
    with open(trajectory_file, 'r') as f:
        data = json.loads(f.read().strip())

    quality = analyze_trajectory(data)

    # 提取关键训练特征
    training_data = {
        'trajectory_id': data['turn_id'],
        'user_input': data['user_input'],
        'messages': data['messages'],
        'sub_turns': data['sub_turns'],
        'tokens': data['tokens'],  # v2 token layer
        'quality_score': quality.overall_score,
        'terminal_state': data['result']['terminal_state'],
        'wall_time': data['result']['wall_secs'],

        # RL奖励信号
        'reward_components': {
            'task_completion': quality.task_completion,
            'efficiency': quality.tool_usage_efficiency,
            'error_recovery': quality.error_recovery,
            'relevance': quality.response_relevance
        },

        # 训练标签
        'should_use_tools': any(word in data['user_input'].lower()
                               for word in ['命令', 'shell', '文件', '目录', 'command', 'file']),
        'error_patterns': [m['content'] for m in data['messages']
                          if m['role'] == 'tool' and '[stderr]' in m['content']],

        # 元数据
        'model': data['model_id'],
        'backend': data['backend'],
        'schema_version': data['schema_version']
    }

    return training_data

def main():
    print("🤖 ARLE轨迹质量评估报告")
    print("=" * 60)

    files = glob.glob('*.jsonl')
    training_dataset = []

    for file in sorted(files):
        print(f"\n📊 分析: {file}")

        with open(file, 'r') as f:
            data = json.loads(f.read().strip())

        quality = analyze_trajectory(data)
        training_data = prepare_training_data(file)
        training_dataset.append(training_data)

        print(f"  质量评分: {quality.overall_score:.2f}/1.00")
        print(f"  - 任务完成: {quality.task_completion:.2f}")
        print(f"  - 工具效率: {quality.tool_usage_efficiency:.2f}")
        print(f"  - 错误恢复: {quality.error_recovery:.2f}")
        print(f"  - 响应相关: {quality.response_relevance:.2f}")

        # 训练建议
        if quality.overall_score < 0.3:
            print("  💡 建议: 低质量轨迹，适合负面示例训练")
        elif quality.overall_score > 0.8:
            print("  ⭐ 建议: 高质量轨迹，适合正面强化")
        else:
            print("  📚 建议: 中等质量，适合改进学习")

    # 保存训练数据集
    with open('training_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n📋 训练数据总结:")
    print(f"  总轨迹数: {len(training_dataset)}")
    print(f"  平均质量: {sum(t['quality_score'] for t in training_dataset) / len(training_dataset):.2f}")
    print(f"  v2 Schema: {sum(1 for t in training_dataset if t['schema_version'] == 2)}/{len(training_dataset)}")
    print(f"  已保存到: training_dataset.json")

if __name__ == "__main__":
    main()