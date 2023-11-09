from PIL import Image
import os
import imagehash
def getCenter(im):
    width, height = im.size  # Get dimensions
    new_width, new_height = 55, 55
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.show()


def getCorners(im,file_name="",path=""):
    size = (40,40)

    width, height = im.size  # Get dimensions


    corners = [
        (0, 0, 0+size[0], 0+size[1] ), #1

        (width-size[0] , 0 , width, size[1]), # 2

        (0,height-size[1], size[0], height), #3

        (width-size[0], height-size[1], width, height), # 4

    ]
    i=0
    for corner in corners:
        i+=1
        this = im.crop(corner)
        this.save(f"{path}{file_name}_{i}.png")




im = Image.open("example.png")
getCorners(im)



if 0:
    for file in os.listdir("split"):
        im = Image.open(f"split/{file}")
        getCorners(
            im,
            file_name = imagehash.phash(im), # unique names per Perceptual hash.
            path="SQCIR/"
        )
