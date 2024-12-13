from openai import OpenAI
import os
import json
from tqdm import tqdm
from utils import load, standardize_bbox, color_map
from render import render
from types import SimpleNamespace
from utils import *


USE_GPT=False

images_dir = '/Users/jinjiahe/Desktop/学习与课程/大三/计算机视觉/GPT4Point/data/cap3d/images'
answer_file = ''

if USE_GPT:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
else:
    api_key = "EMPTY"
    base_url = "http://localhost:8002/v1"

model = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

class VLM3D:
    def __init__(self):
        self.model=model
        self.model_name=model.models.list().data[0].id


    def inferece(self, prompt, point_file):
        image_path = self.render(point_file)
        base64_image = encode_image(image_path)
        messages = get_mllm_messages(prompt, base64_image)
        completion = model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.8
        )
        return completion.choices[0].message.content

    def render(self, point_file):
        # 确保输出目录存在
        os.makedirs(images_dir, exist_ok=True)
        # 构建图片路径
        image_file_name = os.path.splitext(os.path.basename(point_file))[0] + '.png'
        image_path = os.path.join(images_dir, image_file_name)
        # 如果图片不存在
        if not os.path.exists(image_path):
            # 创建配置对象
            config = SimpleNamespace(
                workdir='temp_render',  # 临时工作目录
                output=image_path,  # 输出文件名
                path=point_file,  # 输入文件名
                res=[768, 768],        # 渲染分辨率
                radius=0.025,          # 点的半径
                contrast=0.0004,       # 对比度
                type="point",          # 渲染类型
                view=[2.75, 2.75, 2.75],  # 视角位置
                translate=[0, 0, 0],   # 平移参数
                scale=[1, 1, 1],       # 缩放参数
                white=False,            # 使用白色渲染
                RGB=[],                # RGB颜色设置（空表示使用默认）
                rot=[],                # 旋转参数（空表示使用默认）
                median=None,           # 中值滤波
                separator=",",         # 文本分隔符
                mask=False,            # 点云遮罩
                bgr2rgb=False,         # BGR 转 RGB
                single_view=False,     # 单视图点云
                upsample=1,            # 点云上采样
                num=float('inf'),      # 下采样点数
                knn=False,             # 是否使用KNN颜色映射
                center_num=24,         # KNN中心点数量
                part=False,            # 是否进行KNN聚类并分段渲染
                render=True,           # 使用mitsuba渲染
                tool=False,            # 是否使用实时点云可视化工具
                bbox='none'            # 实时工具边界框可视化
            )
        
            # 加载点云数据
            pcl = load(point_file, separator=",")
            # 标准化点云
            pcl, center, scale = standardize_bbox(config, pcl)
            # 设置点云颜色
            pcl = color_map(config, pcl)
            # 渲染点云
            render(config, pcl)
            print(f"Rendered {point_file} to {image_path}")
        else:
            print(f"Image {image_path} already exists")
        return image_path
    
if __name__ == "__main__":
    point_file = '/Users/jinjiahe/Desktop/学习与课程/大三/计算机视觉/GPT4Point/data/cap3d/points_eval/Cap3D_pcs_8192_xyz_w_color/0a1be4094d844d72b225de98da809b02.pkl'
    vlm3d = VLM3D()
    prompt = "This is a point cloud of"
    print(vlm3d.inferece(prompt, point_file))

