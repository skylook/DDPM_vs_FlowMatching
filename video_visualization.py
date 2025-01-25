#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from moviepy.editor import ImageSequenceClip
import argparse
import sys

def images_to_video(input_dir, fps, output_filename="output_video.mp4"):
    print(f"输入文件夹: {input_dir}")
    if not os.path.isdir(input_dir):
        print(f"错误: 输入路径 '{input_dir}' 不是一个有效的文件夹。")
        sys.exit(1)
    
    # 获取所有图片文件并按文件名排序
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    # 根据文件的修改时间进行排序，os.path.getmtime返回文件的最后修改时间
    images.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))
    print(f"找到 {len(images)} 张图片。")
    
    if not images:
        print("指定文件夹下没有找到任何图片文件。")
        return
    
    # 构建图片的完整路径列表
    image_paths = [os.path.join(input_dir, img) for img in images]
    
    try:
        # 创建视频剪辑
        clip = ImageSequenceClip(image_paths, fps=fps)
        
        # 定义输出路径
        output_path = os.path.join(input_dir, output_filename)
        print(f"输出视频路径: {output_path}")
        
        # 写入视频文件
        clip.write_videofile(output_path, codec='libx264')
        print(f"视频已生成: {output_path}")
    except Exception as e:
        print(f"错误: 无法生成视频, 错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="将指定文件夹下的所有图片合成为视频。")
    parser.add_argument("--input_dir", type=str, required=True, help="图片所在的文件夹路径")
    parser.add_argument("--fps", type=float, default=5.0, help="输出视频的帧率")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="输出视频的文件名")
    args = parser.parse_args()

    images_to_video(args.input_dir, args.fps, args.output)

if __name__ == "__main__":
    main()
