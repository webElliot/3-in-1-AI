from PIL import Image


def getCorners(im):
    size = (40,40)
    width, height = im.size  # Get dimensions
    corners = [
        (0, 0, 0+size[0], 0+size[1] ), #1
        (width-size[0] , 0 , width, size[1]), # 2
        (0,height-size[1], size[0], height), #3
        (width-size[0], height-size[1], width, height), # 4
    ]
    i=0
    each_corner = {}
    for corner in corners:
        i+=1
        this = im.crop(corner)
        each_corner[i]=this
        #this.save(f"{path}{file_name}_{i}.png")
    return each_corner


