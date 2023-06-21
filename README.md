# EasyLabel
EasyLabel is a python script to make data labelling easier. Though software similar to this exists it was just easier to make one. It is here for the community if anyone wishes to label data without hassle

<img src="Example assets/image.png" width="25%">

## Labelling data
Firstly we want to call in a dataset. This has been set up to read image files using cv2. 
```
import EasyLabel as el

label=el.plotter()
label.load_dataset("/Example video/videoName.avi")
```
Now we can plot the data and label it. Left click to label and right click to undo. Once you have fully labelled your image press q. To save the data gathered as current you will need to do ```p.save(filename)```. 

### Normal label
Normal label requires you to click every point. This is useful if your frames are not in any sort of temporal relationship.

```
#label half the dataset
for i in range(0,len(label.images)//2):
    label.plot(i)
    label.save("filename1")
```

### Auto label
If you have frames from a video where each frame relates, you may find a lot of points do not change. This method lets you set the points, and then these points carry through to each frame. To replace a point, press near it and the pont becomes the new when you press q. After laying your initial points, you do not have to do it in any particular order - the function maps points to the closest new point. 

```
#label half the dataset
for i in range(0,len(label.images)//2):
    label.autoplot(i)
    label.save("filename1")
```

## Merging data
If you label 10 points in image one and 11 in image two these will not match and you will get an assertion error. This will require you to start from image two and make a new dataset. Then you will want to merge them. After loading your video file, you can select the files and from waht indicies to save data.

```
label.merge(["filename1.npy","filename2.npy"],[0,1],"new_filename")
```
This will save the filename as "new_filename" and also create a numpy file "new_filenameimages" to store all the images in order.

## Augmentation
You can augment the images by translating them in the x and y, and changing the light intensity. This will generate a dataset of size x*n where x is the number in the dataset and n is the number you set the augmentation.

```
n=10
label.augment("new_filename_","new_filenameimages.npy",n,"augmented")
```
