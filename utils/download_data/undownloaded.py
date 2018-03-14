import os
import sys
import json
from download_data import ParseData

def undownloaded_images(jsonFile, saved_dir):
    """
    Args:
        jsonFile: json file provided by the competition organiser.
        saved_dir: the directory for saveing images.

    Returns:
        A dict with the same structure as jsonFile.
    """
    
    # Cache the downloaded image file names.
    downloaded = {}
    for dirpath, dirnames, filenames in os.walk(saved_dir):
        downloaded.update(zip(filenames, range(len(filenames))))

    # Get the key_url list from jsonFile
    key_url_list = ParseData(jsonFile)

    # Transfrom key_url to image file name
    fn_url_iter = map(lambda key_url: ('%s.png' % key_url[0], key_url[1]), key_url_list)

    # Traverse the image_filenames to find the undownloaded images
    undownloaded = {}
    images = []
    annotations = []
    for fn, url in fn_url_iter:
        if fn not in downloaded:
            mp, ext = os.path.splitext(fn)
            iid_lid = mp.split('_')
            try:
                iid, lid = iid_lid
            except:
                iid, lid = iid_lid[0], None
            
            iid and images.append({'image_id': iid, 'url': [url]})
            lid and annotations.append({'image_id': iid, 'label_id': lid})

    # images and undownloaded['images'] = images
    if images:
        undownloaded['images'] = images
    # annotations and undownloaded['annotations'] = annotations
    if annotations:
        undownloaded['annotations'] = annotations

    return undownloaded


def save_undownloaded_to_jsonfiles(jsonFile, saved_dir, out_json):
    undownloaded = undownloaded_images(jsonFile, saved_dir)
    with open(out_json, 'w') as f:
        json.dump(undownloaded, f)


if __name__ == '__main__':
    # jsonFile = 'validation.json'
    # saved_dir = '/home/huayun/linan/Projects/iMaterialist/data_set/validation'
    # res = undownloaded_images(jsonFile, saved_dir)
    # print(res, len(res['images']))
    if len(sys.argv) != 4:
        print('Syntax: %s <train|validation|test.json> <output_dir> <out_json>' % sys.argv[0])
        sys.exit(0)

    (jsonFile, saved_dir, out_json) = sys.argv[1:]
    save_undownloaded_to_jsonfiles(jsonFile, saved_dir, out_json)
