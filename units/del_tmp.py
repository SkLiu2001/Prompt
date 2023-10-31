import os


def delete_files_in_directory(directory_path):
    # 列出目录下的所有文件和子目录
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        # 判断是否是文件
        if os.path.isfile(file_path):
            # 删除文件
            os.remove(file_path)
        # 判断是否是目录
        elif os.path.isdir(file_path):
            # 递归删除子目录中的文件
            delete_files_in_directory(file_path)
            # 删除空目录
            os.rmdir(file_path)
