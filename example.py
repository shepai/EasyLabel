import EasyLabel as el

label=el.plotter()
label.load_dataset("/Example assets/push.avi")

#label half the dataset
for i in range(0,len(label.images)//2):
    label.autoplot(i)
    label.save("filename1")
#label the other half
for i in range(len(label.images)//2,len(label.images)):
    label.autoplot(i)
    label.save("filename2")

#merge and aurgment
label.merge(["filename1.npy","filename2.npy"],[0,len(label.images)//2],"new data")
label.augment("new data","new dataimages.npy",10,"augmented")
