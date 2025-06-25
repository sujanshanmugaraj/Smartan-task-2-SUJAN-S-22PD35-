# import os
# import re

# # Set this to your yolov5 root folder
# BASE_DIR = 'yolov5'

# # Fix mapping
# fix_patterns = [
#     (r'from utils\.utils', 'from utils'),
#     (r'from yolov5\.utils\.utils', 'from yolov5.utils'),
#     (r'from models\.utils', 'from models'),
#     (r'from \.utils\.utils', 'from .'),
#     (r'from \.utils\.', 'from .'),  # collapse extra .utils.
# ]

# def fix_imports_in_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     original = content
#     for pattern, replacement in fix_patterns:
#         content = re.sub(pattern, replacement, content)

#     if content != original:
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(content)
#         print(f"✅ Fixed: {file_path}")

# def recursively_fix_imports(base_dir):
#     print(f"🔧 Scanning directory: {base_dir}")
#     for root, _, files in os.walk(base_dir):
#         for filename in files:
#             if filename.endswith('.py'):
#                 full_path = os.path.join(root, filename)
#                 fix_imports_in_file(full_path)

# if __name__ == '__main__':
#     recursively_fix_imports(BASE_DIR)
#     print("🎉 All applicable import paths patched successfully!")


import os

file_path = 'yolov5/utils/general.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(file_path, 'w', encoding='utf-8') as f:
    for line in lines:
        if 'from utils.general import colorstr' in line:
            f.write('from .general import colorstr\n')
        else:
            f.write(line)

print("✅ Fixed circular import in general.py")
