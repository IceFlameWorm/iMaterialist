import os
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
    annotation = []
    for fn, url in fn_url_iter:
        if fn not in downloaded:
            mp, ext = os.path.splitext(fn)
            iid_lid = mp.split('_')
            try:
                iid, lid = iid_lid
            except:
                iid, lid = iid_lid[0], None
            
            iid and images.append({'image_id': iid, 'url': url})
            lid and annotation.append({'image_id': iid, 'label_id': lid})

    # images and undownloaded['images'] = images
    if images:
        undownloaded['images'] = images
    # annotation and undownloaded['annotation'] = annotation
    if annotation:
        undownloaded['annotation'] = annotation

    return undownloaded

if __name__ == '__main__':
    jsonFile = 'validation.json'
    saved_dir = '/home/huayun/linan/Projects/iMaterialist/data_set/validation'
    res = undownloaded_images(jsonFile, saved_dir)
    print(res, len(res['images']))
