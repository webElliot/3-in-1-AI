from PIL import Image
import os

imageSize = 98,98

top_left = [
    (1,1),
    (1,100),

    (100,1),
    (100,100),

    (200,1),
    (200,100),
]

dimensions = [
    (each[0],each[1] , each[0]+imageSize[0],each[1] +imageSize[0], )
    for each in top_left
]



i=0
for file in os.listdir("challenges"):
    if "animalLookStanding" in file:
        this_img = Image.open(f"challenges/{file}")
        #this_img.save(f"animalLookStanding/{file.split('_')[1]}")
        n=0
        for each in dimensions:
            num = this_img.crop(each)
            n += 1
            num.save(f"split/{n}_{file.split('_')[1]}")

        i+=1

print(f"{i} Images saved.")