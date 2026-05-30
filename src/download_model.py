import os
from modelscope.hub.snapshot_download import snapshot_download
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

from src.utils.tools import get_project_root
def download_qwen_model():
    root_dir = get_project_root()
    # 动态设定模型保存路径
    model_dir = os.path.join(root_dir, "models", "pretrained")
    os.makedirs(model_dir, exist_ok=True)
    
    # 🔴 升级为 30亿 参数的 Qwen2.5-3B-Instruct
    model_id = 'qwen/Qwen1.5-0.5B-Chat'
    # 
    # qwen/Qwen2.5-3B-Instruct
    
    print(f"🚀 开始从 ModelScope 下载大模型 {model_id}...")
    print(f"📂 模型将保存在: {model_dir}")
    print("⏳ 模型文件较大，Kaggle 环境下预计需要 1-2 分钟，请稍候...")
    
    # 执行下载
    model_path = snapshot_download(
        model_id, 
        cache_dir=model_dir,
        revision='master'
    )
    
    print(f"✅ 模型下载完成！本地绝对路径为: {model_path}")
    print("💡 请记下这个文件夹名称，下一步我们需要把它填进训练脚本中。")

if __name__ == "__main__":
    download_qwen_model()