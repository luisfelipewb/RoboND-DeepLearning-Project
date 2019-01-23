from PIL import Image
import os, fnmatch



path_list = [ '../data/train/images/',
              '../data/train/masks/',
              '../data/validation/images/',
              '../data/validation/masks/' ]


pattern1 = "*.png"
pattern2 = "*.jpeg"

def flip_image(image_path, save_path):
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(save_path)


for path in path_list:
    image_list = os.listdir(path)
    image_number = 0

    for image in image_list:
        if fnmatch.fnmatch(image, pattern1) or fnmatch.fnmatch(image, pattern2) :
            #print (image)
            image_path = path + image
            new_image_path = path + 'flipped_' + image
            flip_image(image_path, new_image_path)
            image_number += 1

    print ("{} generated images in {}".format(image_number, path))

