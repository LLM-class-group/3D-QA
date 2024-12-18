import os
import argparse
from types import SimpleNamespace
from utils import load, standardize_bbox, color_map, rotation
from render import render
import numpy as np
import time
input_dir = ''
images_dir = ''


def render_point_cloud_to_image(point_file, images_dir, BEV=False):
    # 确保输出目录存在
    os.makedirs(images_dir, exist_ok=True)
    # 构建输出路径
    output_file_name = os.path.splitext(os.path.basename(point_file))[0] + '.png'
    output_path = os.path.join(images_dir, output_file_name)
    # 如果图片已存在则直接返回
    if os.path.exists(output_path):
        return output_path

    # 创建配置对象
    config = SimpleNamespace(
        workdir='temp_render2',  # 临时工作目录
        output=output_path,  # 输出文件名
        path=point_file,  # 输入文件名
        res=[768, 768],        # 渲染分辨率
        radius=0.015,          # 点的半径
        contrast=0.0004,       # 对比度
        type="point",          # 渲染类型
        view=[2.75, 2.75, 2.75],  # 视角位置
        translate=[0, 0, 0],   # 平移参数
        scale=[1, 1, 1],       # 缩放参数
        white=False,            # 使用白色渲染
        RGB=[],                # RGB颜色设置（空表示使用默认）
        rot=[90, 0, 20],        # 旋转参数（x, y, z）
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

    if BEV:
        config.view = [0, 0, 2]
        config.rot = [90, 0, 20]
    
    # 加载点云数据
    pcl = load(point_file, separator=",")
    
    # 标准化点云
    pcl, center, scale = standardize_bbox(config, pcl)
    
    # 旋转点云
    if len(config.rot) != 0:
        assert len(config.rot) == 3
        rot_matrix = rotation(config.rot)
        pcl[:, :3] = np.matmul(pcl[:, :3], rot_matrix)
    
    # 设置点云颜色
    pcl = color_map(config, pcl)
    
    # 渲染点云
    render(config, pcl, BEV)
        
    return output_path

def render_all_point_clouds(point_folder, images_dir, BEV=False):
    # 遍历文件夹中的所有点云文件（倒序）
    num = 0
    for point_file in sorted(os.listdir(point_folder), reverse=True):
        num += 1
        if point_file.endswith('.npy') or point_file.endswith('.pkl') or point_file.endswith('.ply'):
            point_path = os.path.join(point_folder, point_file)
            print(f"正在渲染第{num}个点云: {point_file}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
            # 渲染点云图像
            render_point_cloud_to_image(point_path, images_dir, BEV)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-folder", type=str, default=input_dir)
    parser.add_argument("--image-folder", type=str, default=images_dir)
    parser.add_argument("--BEV", action="store_true")
    args = parser.parse_args()

    render_all_point_clouds(args.point_folder, args.image_folder, args.BEV)