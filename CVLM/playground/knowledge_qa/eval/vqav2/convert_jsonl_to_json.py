#!/usr/bin/env python3
"""
将JSONL文件转换为JSON数组格式，使其与VQA库兼容
"""

import json

def convert_jsonl_to_json(input_file, output_file):
    """
    将JSONL文件转换为JSON数组格式
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSON文件路径
    """
    results = []
    
    print(f"正在读取JSONL文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    results.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析错误: {e}")
                    continue
    
    print(f"成功读取 {len(results)} 条记录")
    print(f"正在写入JSON文件: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("转换完成！")

if __name__ == "__main__":
    input_file = "llava_vqav2_mscoco_val.jsonl"
    output_file = "llava_vqav2_mscoco_val.json"
    
    convert_jsonl_to_json(input_file, output_file) 