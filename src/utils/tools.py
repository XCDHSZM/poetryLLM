import os

def get_project_root():
    """
    动态获取项目根目录
    无论此函数被谁调用，都以 tools.py 所在位置为基准向上推
    """
    # 当前 tools.py 的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 向上推三级：tools.py -> utils/ -> src/ -> 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    return project_root