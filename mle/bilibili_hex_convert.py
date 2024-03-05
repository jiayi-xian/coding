import os
import binascii
import glob, shutil

def process_files(base_directory=None):
    """

    """
    """
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith("30280.m4s"):
            filepath = os.path.join(directory, filename)
        else:
            continue
    """
    # 使用glob模块来查找所有子文件夹中的.m4s文件
    for filepath in glob.glob(os.path.join(base_directory, '*30280.m4s')):
        with open(filepath, 'rb') as file:
            file_data = file.read()
        
        with open(filepath, 'rb') as file:
            file_data = file.read()
        
        # 转换为十六进制，检查并删除指定的十六进制字符串
        hex_data = binascii.hexlify(file_data)
        prefix = b'303030303030303030'
        if hex_data.startswith(prefix):
            hex_data = hex_data[len(prefix):]

        # 创建新的文件名（.mp4）
        new_filename = os.path.splitext(filepath)[0] + '.mp4'

        # 将处理后的数据写入新文件
        with open(new_filename, 'wb') as new_file:
            new_file.write(binascii.unhexlify(hex_data))

def move_mp4_files(source_directory, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Search for .mp4 files in the source directory
    for filepath in glob.glob(os.path.join(source_directory, '*/*.mp4')):
        # Define the new file path in the target directory
        new_filepath = os.path.join(target_directory, os.path.basename(filepath))

        # Move the file to the new directory
        shutil.move(filepath, new_filepath)
        print(f"Moved file: {filepath} to {new_filepath}")

# Source and target directories
source_directory = '/Users/jiayixian/Movies/bilibili/934606801'
target_directory = '/Users/jiayixian/Movies/bilibili/934606801/mp4'

if not os.path.exists(target_directory):
    os.makedirs(target_directory)
    print(f"Directory '{target_directory}' was created.")
else:
    print(f"Directory '{target_directory}' already exists.")

# 使用示例
directory = '/Users/jiayixian/Movies/bilibili/934606801'  # 基础文件夹路径
process_files(directory)

# Execute the function
move_mp4_files(source_directory, target_directory)