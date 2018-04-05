import os
import glob
import argparse
import shutil
from threading import Thread

IMAGE_EXT = 'png'
IMAGE_LABEL_SPLIT = '_'
THREADS = 10
CLASSES = 128

class ImageReorgThread(Thread):
    def __init__(self, image_list, target_path):
        super(ImageReorgThread, self).__init__()
        self._image_list = image_list
        self._target_path = target_path


    def run(self):
        while True:
            try:
                image_file = self._image_list.pop()
            except IndexError:
                return
            
            image_name = os.path.basename(image_file)
            label_id = os.path.splitext(image_name)[0].split(IMAGE_LABEL_SPLIT)[1]
            
            target_path = os.path.join(self._target_path, label_id)
            # 以下代码可能会有冲突,改到在主线程创建文件夹
            # if not os.path.exists(target_path):
            #     os.mkdir(target_path)
            
            shutil.move(image_file, target_path)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="directory containing the image files")
    parser.add_argument('--target_path', type=str, help='directory for containing the classified images')
    args = parser.parse_args()
    image_path = args.image_path
    target_path = image_path if args.target_path is None else args.target_path

    # 获取图片列表
    image_list = glob.glob(os.path.join(args.image_path, '*.' + IMAGE_EXT))

    # 创建以类别命名的文件夹
    for i in range(1, CLASSES + 1):
        label_dir = os.path.join(target_path, str(i))
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
    
    # 启动线程对图片按类别进行存放
    threads = [ImageReorgThread(image_list, target_path) for i in range(THREADS)]
    
    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()