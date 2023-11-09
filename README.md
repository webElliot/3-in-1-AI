# Select the animal sat on a Square looking at the circle
 
**The task was to figure out which image contained the animal sat on X_Shape, looking at Y_Shape**

The following challenge was to find the Lion sat on the white Square looking at the circle.

<img src="https://i.imgur.com/vFGlDoW.png">

This Image detection task required to be broken down in to a few different steps.

# Image detection model

**Important notes** 
- Transfer Learning (Based on the ResNet18 Model)
- Semi-Supervised learning (Getting the AI to predict labels for me after a small dataset had been aquired.)
- The target Accuracy I was aiming for was 99% because Challenges would consist of 1-5 Different 3x2 Images most commonly seen at 5 Images per challenge.
- If all three Models only had a 95% accuracy on each would result in a total accuracy of 85.74% (95/100 ^ 6) for each Challenge image out of the 6. 
- Spread across 5 challenge images at 85.74% Accuracy on each Challenge would result in only a 37% Accuracy per Funcaptcha Task (assuming there were 5 3x2 images you needed to solve correctly)
- Also if you fail funcaptcha Tasks your future tasks get exponentially more difficult to solve (up to 20 Challenge images if you get lots incorrect)
- This is why it is vital to achieve 99-99.9% Accuacy per task

### Shape sat on
[Shape sat Model]("https://github.com/webElliot/3-in-1-AI/tree/main/Shape%20sat%20Model"")
1) First, I downloaded the data for the challenges
2) I wrote a script **cutter.py** to split the image into 6.


<p>
    <img src="https://i.imgur.com/89dva6u.png">
    <img src="https://i.imgur.com/iGWeJyo.png">
    <img src="https://i.imgur.com/iYySnVz.png">
</p>
<p>
    <img src="https://i.imgur.com/cYs2Bfk.png">
    <img src="https://i.imgur.com/OGgZzLV.png">
    <img src="https://i.imgur.com/KwTwxCt.png">
</p>

3) I then imported lots of these images into my [Label Webserver]("https://github.com/webElliot/LabelWS")
4) I Created the Labels: Circle, Square and clicked around 100 images.
5) I then started Training my Circle/Square model
6) After training I would get ~60-75% accuracy, since this is better than guessing I then input my minority class AI Predicted image labels into the webserver & Repeated the labelling process
7) Having images the AI already predicted as the correct class made it much faster to select the correct & Incorrect answers.
8) After a few more repeats the accuracy ended up at ~99.9% with a very small dataset.


### Eye direction 
[Eye direction Model]("https://github.com/webElliot/3-in-1-AI/tree/main/Eye%20direction%20model")
1) The next step was creating Labels for each corner; Top Left, Top Right, Bottom Left, Bottom Right : Labelled as; 1,2,3,4 respectively.
2) Then I began labelling.
3) Training the Eye direction model.
4) After labelling a few I used the technique of getting the AI to make predictions based on it's limited learning to help increase the probability the targetted label was the one I was labelling.
5) This model took a lot more images than the first one, because the Eye images are very different across the Challenges.
6) After labelling 500 of each Label, I stopped labelling & training because accuracy reached 99%.

### Corner Shape Labelling
[Corner shape Model]("https://github.com/webElliot/3-in-1-AI/tree/main/Corner%20shape%20model"")
1) Firstly I used PIL in Python to crop the images just so I had the corners of the images since the main focus was on the green shape.

**corner.py** is the script I wrote to grab the corners of the images.

**This would mean the noise in the image would be reduced since the target shape takes up the majority of the image**

<p>
    
<img src="https://i.imgur.com/IrVH56E.png">
<img src="https://i.imgur.com/O3VhhJj.png">
<img src="https://i.imgur.com/OGfp03a.png">
<img src="https://i.imgur.com/mWb7LWL.png">
<img src="https://i.imgur.com/0qDMYqf.png">
</p>

2) Next I input these images into the Label WS, Began labelling and repeated the process of getting AI predictions and labelling those correctly.
3) This model required a very small dataset to achieve 99.9% Accuracy.
