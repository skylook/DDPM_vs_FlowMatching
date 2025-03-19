#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import sys
from pathlib import Path
import glob
import re

# 自然排序函数
def natural_sort_key(s):
    """
    用于自然排序的键函数
    将字符串中的数字部分转换为整数进行比较
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def images_to_video(input_dir, fps, output_filename="output_video.mp4"):
    print(f"输入文件夹: {input_dir}")
    input_path = Path(input_dir).absolute()
    if not input_path.is_dir():
        print(f"错误: 输入路径 '{input_dir}' 不是一个有效的文件夹。")
        sys.exit(1)
    
    # 获取所有图片文件并按文件名排序
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [str(f.absolute()) for f in input_path.iterdir() 
             if f.suffix.lower() in supported_extensions]
    
    # 使用自然排序
    images.sort(key=natural_sort_key)
    print(f"找到 {len(images)} 张图片。")
    
    # 打印前10个和后10个文件名，用于验证排序
    if images:
        print("排序后的前10个文件:")
        for i, img in enumerate(images[:10]):
            print(f"  {i+1}. {os.path.basename(img)}")
        
        if len(images) > 20:
            print("...")
            
        print("排序后的后10个文件:")
        for i, img in enumerate(images[-10:]):
            print(f"  {len(images)-9+i}. {os.path.basename(img)}")
    
    if not images:
        print("指定文件夹下没有找到任何图片文件。")
        return
    
    try:
        # 定义输出路径
        if os.path.isabs(output_filename):
            output_path = output_filename
        else:
            output_path = str(input_path / output_filename)
        print(f"输出视频路径: {output_path}")
        
        # 创建临时文件列表
        temp_list_file = str(input_path / "temp_file_list.txt")
        with open(temp_list_file, 'w') as f:
            for img_path in images:
                f.write(f"file '{os.path.basename(img_path)}'\n")
                f.write(f"duration {1/fps}\n")
            # 最后一帧也需要持续时间
            f.write(f"file '{os.path.basename(images[-1])}'\n")
            f.write(f"duration {1/fps}\n")
        
        # 使用ffmpeg的concat方式，确保按照文件列表的顺序
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-vsync', 'vfr',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        print("正在生成视频...")
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行ffmpeg命令
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=str(input_path)  # 设置工作目录为输入目录
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"ffmpeg错误: {stderr.decode()}")
            sys.exit(1)
        
        # 删除临时文件
        os.remove(temp_list_file)
        
        print(f"视频已生成: {output_path}")
        
    except Exception as e:
        print(f"错误: 无法生成视频, 错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="将指定文件夹下的所有图片合成为视频。")
    parser.add_argument("--input_dir", type=str, required=True, help="图片所在的文件夹路径")
    parser.add_argument("--fps", type=float, default=5.0, help="输出视频的帧率")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="输出视频的文件名")
    args = parser.parse_args()

    images_to_video(args.input_dir, args.fps, args.output)

if __name__ == "__main__":
    main()
