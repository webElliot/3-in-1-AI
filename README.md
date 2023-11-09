# Select the Lion sat on a Square looking at the circle
 
**The task was to figure out which image contained X_Animal looking at Y_Shape.**

The following challenge was to find the Lion sat on the white Square looking at the circle.

<img src="https://i.imgur.com/vFGlDoW.png">

This Image detection task required to be broken down in to a few different steps.

# Image detection model

### Shape sat on

1) First, I downloaded the data for the challenges
2) I wrote a script **cutter.py** to split the image into 6.


<img src="https://i.imgur.com/89dva6u.png">
<img src="https://i.imgur.com/iGWeJyo.png">
<img src="https://i.imgur.com/iYySnVz.png">
<img src="https://i.imgur.com/cYs2Bfk.png">
<img src="https://i.imgur.com/OGgZzLV.png">
<img src="https://i.imgur.com/KwTwxCt.png">

3) I then imported lots of these images into my [Label Webserver]("https://github.com/webElliot/LabelWS")
4) I Created the Labels: Circle, Square and clicked around 100 images.
5) I then started Training my [Circle/Square Module]("LINKMEPLS"")
6) After training I would get ~60-75% accuracy, since this is better than guessing I then input my minority class AI Predicted image labels into the webserver & Repeated the labelling process
7) Having images the AI already predicted as the correct class made it much faster to select the correct & Incorrect answers.
8) After a few more repeats the accuracy ended up at ~99.9% with a very small dataset.


### Eye direction
1) The next step was creating Labels for each corner; Top Left, Top Right, Bottom Left, Bottom Right
2) Then I began labelling.
3) Training the Eye direction model: [Eye direction Model]("EyeMODEL")
4) After labelling a few I used the technique of getting the AI to make predictions based on it's limited learning to help increase the probability the targetted label was the one I was labelling.
5) This model took a lot more images than the first one, because the Eye images are very different across the Challenges.
6) After labelling a lot of images, I stopped training once accuracy reached 90%.

### Corner Shape Labelling
1) Firstly I used PIL in Python to crop the images just so I had the corners of the images since the main focus was on the green shape.

**corner.py** is the script I wrote to grab the corners of the images.

**This would mean the noise in the image would be reduced since the target shape takes up the majority of the image**

<img src="https://i.imgur.com/IrVH56E.png" width="=50">
<img src="https://i.imgur.com/O3VhhJj.png" width="50">
<img src="https://i.imgur.com/OGfp03a.png">
<img src="https://i.imgur.com/mWb7LWL.png">
<img src="https://i.imgur.com/0qDMYqf.png">

2) Next I input these images into the Label WS, Began labelling and repeated the process of getting AI predictions and labelling those correctly.
3) This model required a fairly small dataset to achieve 95-99% Accuracy.
