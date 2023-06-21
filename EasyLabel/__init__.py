import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import copy

class plotter:
    def __init__(self):
        self.images=[]
        self.points=[]
        self.all_points=[]
        self.ps=[]
    def plot(self,num=0):
        self.points=[]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(str(num))
        self.ax.imshow(self.images[num])
        if len(self.all_points)>0: #past points placed
            p=np.array(self.all_points[-1])
            plt.scatter(p[:,0],p[:,1],c="b",s=1)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
    def autoplot(self,num=0):
        self.points=[]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(str(num))
        self.ax.imshow(self.images[num])
        if len(self.all_points)>0: #past points placed
            p=np.array(self.all_points[-1])
            plt.scatter(p[:,0],p[:,1],c="b",s=1)
        cid = self.fig.canvas.mpl_connect('key_press_event', self.button)
        cid2 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
    def button(self,event): #allow event based clicking
        print("event")
        if str(event.key)=="enter" or str(event.key)=="q":
            if len(self.all_points)>0: #past points placed
                p=np.array(self.all_points[-1])
                if len(self.points)<=0: #save previous
                    self.points=p.copy()
                elif len(self.points)==len(self.all_points[0]):
                    pass
                else: #find closest with each point
                    points=p.copy()
                    for point in self.points: #find distance
                        dist = np.linalg.norm(point-points,axis=1)
                        points[np.argmin(dist)]=point #assign closest
                    self.points=points.copy()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def onclick(self,event): #allow event based clicking
        global ix, iy
        ix, iy = event.xdata, event.ydata
        if str(event.button)==str("MouseButton.LEFT"): #add if left click
            coords = [ix, iy]
            self.points.append(coords)
            self.ps.append(self.ax.scatter(coords[0],coords[1]))
        elif str(event.button)==str("MouseButton.RIGHT") and len(self.points)>0: #remove if right click
            self.points.pop(len(self.points)-1)
            self.ps[-1].remove()
            self.ps.pop(len(self.ps)-1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def load_dataset(self,file,every=5):
        """
        @param file is the file path to the video file
        @param every is take every frames. Set to 1 for whole dataset
        """
        cap=cv2.VideoCapture(file)
        p=cv2.imread(file)
        ret = True
        i=0
        while ret: #while frames
            ret, frame = cap.read()
            if i%every==0 and ret: 
                self.images.append(frame)
            i+=1
        cap.release()
        print("DATASET SIZE:",len(self.images))
    def save(self,filepath="test3"):
        np.save(filepath,np.array(self.all_points))
        if len(self.all_points)>0: 
            print(len(self.points),len(self.all_points[0]))
            assert len(self.points)==len(self.all_points[0]), "Arrays not the same length"
        self.all_points.append(self.points)
    def view(self,filename,dataset_num,i=0,mode=0):
        """
        @param filename to load points
        @param i is the starting position
        @param dataset_num is the starting position in images, if you give a string filename it will load that dataset
        """
        plt.cla()
        im=[]
        if type(dataset_num)==type(""): #if the number is actually a sub image set
            im=np.load(dataset_num)
        else: im=self.images.copy()
        points=np.load(filename)
        p=points[i]
        if mode==0:
            plt.scatter(p[:,0],p[0:,1])
        elif mode==1:
            for j,_ in enumerate(p):
                cv2.putText(im[i],str(j),(int(_[0]),int(_[1])),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0))
        plt.imshow(im[i])
        plt.pause(0.1)
    def merge(self,files,images,newfilename):
        """
        merge seperate datasets int one
        @param files is an array of filenames
        @param images is an array of integers [image from]
        @newfilename is the new name to save the merged files
        """
        ar=[]
        images_=[]
        for i,filename in enumerate(files):
            points=np.load(filename)
            print(len(points))
            #print(points.shape,len(self.images[images[i]:len(points)]),images[i],len(points))
            if len(ar)==0: #if first one
                ar+=list(points)
                images_+=self.images[images[i]:images[i]+len(points)]
            elif len(points[0])==len(ar[0]): #if next images
                ar+=list(points)
                images_+=self.images[images[i]:images[i]+len(points)]
            else: print("Not same format",i,len(points[0]))
        ar=np.array(ar)
        images_=np.array(images_)
        for i in range(len(images_)):
            images_[i]=self.saturate(images_[i])
        print("DATASET LENGTH:",ar.shape,images_.shape)
        np.save(newfilename,ar)
        np.save(newfilename+"images",images_)
    def saturate(self,frame):
        lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return frame
    def augment(self,filname,images,make_sampels,new_file):
        """
        augment the data given a filename and images
        @param filename of points
        @param images is filename to images
        @param make_samples is how many new samples we want
        @param new_file is the filename for new augmented data
        """
        images=np.load(images)
        points=np.load(filname)
        new_data=[]
        new_points=[]
        for j in range(make_sampels):
            for i,im in enumerate(images):
                #get random translation
                im=copy.deepcopy(images[i])
                im=self.saturate(im)
                x_shift=random.randint(-30,30)
                y_shift=random.randint(-30,30)
                brightness=random.randint(-15,15)
                im=im-brightness #change brightness
                im[im<0]=0
                im[im>255]=255
                M=np.float32([
                    [1,0,x_shift],
                    [0,1,y_shift]
                ])#matix for movement
                p=copy.deepcopy(points[i])
                shifted=cv2.warpAffine(im,M,(im.shape[1],im.shape[0])) #shift
                p[:,1]=p[:,1]+y_shift
                p[:,0]=p[:,0]+x_shift
                new_points.append(copy.deepcopy(p))
                new_data.append(copy.deepcopy(shifted))
        #save all the data
        new_points=np.array(new_points)
        new_data=np.array(new_data)
        np.save(new_file,new_points)
        np.save(new_file+"images",new_data)
    def mergeImage(self,pointnames,imagenames,name="all"):
        images=None
        points=None
        for i in range(0,len(pointnames)-1):
            if i==0:
                points=np.load(pointnames[i])
                images=np.load(imagenames[i])
            next=np.load(pointnames[i+1])
            next_i=np.load(imagenames[i+1])
            images=np.concatenate((images,next_i))
            points=np.concatenate((points,next))
        print(images.shape,points.shape)
        np.save(name,points)
        np.save(name+"images",images)
p=plotter()
p.load_dataset("C:/Users/dexte/github/RoboSkin/Assets/Video demos/test.avi")
path="C:/Users/dexte/OneDrive/Documents/AI/Data_Labeller/files/"


##########Create datasets###############################
#p.merge([path+"test3_10a.npy",path+"test3_22.npy"],[10,22],path+"new_move")
#p.merge([path+"testnew_tactip0.npy",path+"testnew_tactip5.npy",path+"testnew_tactip10.npy",path+"testnew_tactip15.npy"],[0,5,10,15],path+"new_tactip")
p.merge([path+"testnew_tactip15.npy"],[15],path+"new_tactip")
p.augment(path+"new_tactip.npy",path+"new_tactipimages.npy",10,path+"augmented")
#p.merge([path+"testnew_tactip5.npy",path+"testnew_tactip10.npy"],[0,5],path+"new_tactip_multi")
##########merge datasets################################
#p.mergeImage([path+"new_tactip.npy",path+"new_move.npy"],[path+"new_tactipimages.npy",path+"new_moveimages.npy"],path+"all")
#p.augment(path+"all.npy",path+"allimages.npy",10,path+"augmented2")
"""print(len(np.load(path+"new_tactip.npy")))
for i in range(0,len(np.load(path+"new_tactip.npy"))):
    p.view(path+"new_tactip.npy",path+"new_tactipimages.npy",i=i,mode=0)
    plt.pause(0.5)#"""

#left on 50 ffrom movment3
"""for i in range(15,len(p.images)):
    p.autoplot(i)
    p.save()

#print(p.all_points)#"""

#"""