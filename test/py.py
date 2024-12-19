import json

def print_json_keys(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if isinstance(data, dict):
            # 如果顶层是字典，直接打印键
            for key in data.keys():
                print(key)
        elif isinstance(data, list):
            # 如果顶层是列表，检查每个元素
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    print(f"Keys in element {i}: {list(item.keys())}")
                else:
                    print(f"Element {i} is not a dictionary, it is of type {type(item).__name__}")
        else:
            print(f"Unsupported JSON structure: {type(data).__name__}")

# 示例调用
print_json_keys('d:/project/llama2_cpp/data/data00.json')
