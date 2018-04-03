from PIL import Image
from glob import glob
import os


def trans_png2jpg(img_root, new_img_root):
    if not os.path.exists(new_img_root):
        os.mkdir(new_img_root)
    for imgfile in os.listdir(img_root):
        new_file_name = os.path.join(new_img_root, imgfile.replace('png', 'jpg'))
        if not os.path.exists(new_file_name):
            print(new_file_name)
            try:
                pil_image = Image.open(img_root + '/' + imgfile, 'r')
            except ValueError as VE:
                if str(VE) == 'Decompressed Data Too Large':
                    print('ValueError:Decompressed Data Too Large ,failed to save file %s' % new_file_name)
                continue
            except IOError:
                print('IOError: failed to open file %s' % imgfile)
                continue
            try:
                pil_image = pil_image.convert('RGB')
                pil_image.save(new_file_name, format='JPEG', quality=90)
            except:
                print('error:,failed to save file %s' % new_file_name)
        else:
            print('file exists, skip')


ROOT_PATH = '/media/data2/lzhang_data/dataset/iMaterial/train/'
NEW_ROOT_PATH = '/media/data2/lzhang_data/dataset/iMaterial/train2/'
trans_png2jpg(ROOT_PATH, NEW_ROOT_PATH)
