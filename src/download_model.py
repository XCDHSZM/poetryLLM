import os
from modelscope.hub.snapshot_download import snapshot_download
def get_project_root():
    current_script_path = os.path.abspath(__file__)
    # 根据你存放此脚本的位置向上推，这里假设脚本在 src/ 目录下
    return os.path.dirname(os.path.dirname(current_script_path))

def download_qwen_model():
    root_dir = get_project_root()
    # 动态设定模型保存路径
    model_dir = os.path.join(root_dir, "models", "pretrained")
    os.makedirs(model_dir, exist_ok=True)
    
    # 这里我们先以 Qwen1.5-0.5B 为例，参数量小，练手极快，普通单卡即可跑通全流程
    model_id = 'qwen/Qwen1.5-0.5B-Chat'
    
    print(f"开始从 ModelScope 下载模型 {model_id}...")
    print(f"模型将保存在: {model_dir}")
    
    # 执行下载
    model_path = snapshot_download(
        model_id, 
        cache_dir=model_dir,
        revision='master'
    )
    
    print(f"✅ 模型下载完成！本地绝对路径为: {model_path}")

if __name__ == "__main__":
    download_qwen_model()