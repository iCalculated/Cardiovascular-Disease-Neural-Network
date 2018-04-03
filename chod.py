from tkinter import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
score=0
x=[]
y=[]

# load cleveland heart disease dataset
dataset = np.loadtxt("cleveland_hd.csv", delimiter=",")

#names of the attributes
names=["Age (years)","Sex (1=male, 0=female)", "Reported chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)","Resting blood pressure (mm Hg)","Serum cholesterol (mg/dL)","Does fasting blood sugar exceed 120 mg/dL threshold? (1=yes, 0=no)","Resting electrocardiographic testing results (0=normal, 1=having ST-T wave abnormality, 2=showing probable or definite left ventricular hypertrophy)","Maximum heart rate achieved (bpm)","Exercise-induced angina? (1=yes, 0=no)","ST depression induced by exercise relative to rest","Slope of the peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)","Number of major vessels (0-3) colored by flourosopy"]

#creates the Neural Network
def createNN(X,Y):      
    global score   
    model = Sequential()

    #layers: 4 -> 1000 -> 50 -> 1
    model.add(Dense(1000, input_dim=4, activation='relu'))
    model.add(Dense(50, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #training loop for Neural Network
    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

    #tests Neural Network
    scores = model.evaluate(X, Y)

    #if the score is an error value, redo the trial.
    if scores[1]>0.5385 and scores[1]<0.5390:
        print("    "+"Error occured. Rerunning trial.")
        return 1
    else:
        score=(scores[1]*100)
        return 0

def trial(X,Y):

    #this loop facilitates the rerunning of trials
    succ=createNN(X, Y)
    if succ==1:
        trial(X,Y)

def userInput(X,Y):
    #creates a user input thing
    global predicting 
    predicting = True
    prediction = model.predict(self, x, batch_size=10, verbose=0, steps=None)
    print(prediction)
        if prediction:
        global predicted
        predicted = True
  


#creating the GUI Window
root = Tk()

#Makes the window fullscreen
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.focus_set()

#Pressing escape exits the program
root.bind("<Escape>", lambda e: root.quit())

#Title at the top of GUI
titlel = Label(root, text="Heart Disease Neural Network Simulation", fg="#383a39", bg="#a1dbcd", font=("Helvetica", 40)) 
titlel.pack()

#Our names
names = Label(root, text='''Mani Chadaga, Akshay Nambudripad, and Alexander "Sasha" Hydrie. Ninth Graders at Central High School''' , fg="#383a39", bg="#a1dbcd", font=("Helvetica", 24)) 
names.pack()

#Set background color
root.configure(background="#a1dbcd")

#Initalize the ten photos for the button
p1=PhotoImage(file="p1PNG.png")
p2=PhotoImage(file="p2PNG.png")
p3=PhotoImage(file="p3PNG.png")
p4=PhotoImage(file="p4PNG.png")
p5=PhotoImage(file="p5PNG.png")
p6=PhotoImage(file="p6PNG.png")
p7=PhotoImage(file="p7PNG.png")
p8=PhotoImage(file="p8PNG.png")
p9=PhotoImage(file="p9PNG.png")
p10=PhotoImage(file="p10PNG.png")
reset=PhotoImage(file="reset.png")
relaunch=PhotoImage(file="relaunch.png")

#All labels in the gui
atttoptext=Label(root, font=("Helvetica", 24),fg="#383a39", bg="#a1dbcd",text="Attributes:                ")
atttoptext.pack()
atttoptext.place(x=800,y=650)

step1=Label(root, font=("Helvetica", 18), bg="#a1dbcd",text="1) Choose 2 Attributes:", fg='red')
step1.pack()
step1.place(x=800,y=150)

step2=Label(root, font=("Helvetica", 18), bg="#a1dbcd",text="2) Launch Neural Network", fg='red')
step2.pack()
step2.place(x=1200,y=150)

launchtext=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="Status: Not Launched",width=37)
launchtext.pack()
launchtext.place(x=1140,y=330)

initext=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",width=37)
initext.pack()
initext.place(x=1140,y=380)

initext2=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",width=37)
initext2.pack()
initext2.place(x=1140,y=430)

initext3=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",width=37)
initext3.pack()
initext3.place(x=1140,y=480)

initext4=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",width=37)
initext4.pack()
initext4.place(x=1140,y=530)

initext5=Label(root, font=("Helvetica", 24), bg="#a1dbcd",text="",width=21)
initext5.pack()
initext5.place(x=1140,y=580)

initext6=Label(root, font=("Helvetica", 24), bg="#a1dbcd",text="",width=21)
initext6.pack()
initext6.place(x=1140,y=630)

initext7=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",width=37)
initext7.pack()
initext7.place(x=1140,y=690)

atts=Label(root,fg="#383a39", font=("Helvetica", 11), bg="#a1dbcd",text="",width=28, anchor='w', justify='left')
atts.pack()
atts.place(x=800,y=683)

warntext=Label(root, font=("Helvetica", 14), bg="#a1dbcd",text="",fg="#932d0e")
warntext.pack()
warntext.place(x=755,y=780)

#Initalize various variables
att=[]
launched=False
executed=False
training=False
trained=False
testing=False
accdisplayed=False
queried=False
inputs=[0,1]

#A while True loop in essence.
def launchloop():

    #initalized global variables
    global X
    global Y
    global executed
    global score
    global training
    global trained
    global testing
    global accdisplayed
    global queried
    global predicting

    #The 'text console'
    if launched and executed and training and trained and testing and accdisplayed and queried:
        time.sleep(2)
        initext7['text']="Would you like to relaunch for these\nattributes? Or reset program completely?"
    if launched and executed and training and trained and testing and accdisplayed and not queried:
        time.sleep(1)
        initext5['text']='Accuracy:'
        initext6['text']=str(score)+"%"
        queried=True
    if launched and executed and training and trained and testing and not accdisplayed:
        time.sleep(2)
        initext4['text']='NN Testing Complete.'
        accdisplayed=True
    if launched and executed and training and trained and not testing:
        time.sleep(2)
        initext4['text']='Testing NN Diagnosis Accuracy...'    
        testing=True
    if launched and executed and training and not trained:
        trial(X,Y)
        initext3['text']='NN Self-Training Complete.'
        trained=True
    if launched and executed and not training:
        time.sleep(1) 
        initext3['text']='NN Self-Training Commenced...'
        training=True
    if launched and not executed:
        initext['text']='Dataset Initialized'
        X = dataset[:,inputs]
        Y = dataset[:,13]
        for k in range(len(Y)):
            if Y[k]!=0:
                Y[k]=1  
        time.sleep(1)        
        initext2['text']='Neural Network Initialized'          
        scoreslist=[]
        executed=True


    #This makes it a loop
    root.after(1000,launchloop)    

def update(maxr=0):
    if launched:
        atts['fg']='blue'
    else:
        atts['text']=""
        for i in range(len(att)):
            atts['text']=atts['text']+"\n ● "+att[i]
        if maxr==1:
            warntext['text']="Maximum number of attributes reached!\nYou may now launch the Neural Network.\nOr, remove an attribute by reclicking it."
        else:
            warntext['text']=""

def c1(addendum,o):
    if not launched:
        n=o+1
        if n not in inputs:
            if len(att)>1:
                update(1)
            else:
                inputs.append(n)
                att.append(addendum)
                update()
        else:
            att.remove(addendum)
            inputs.remove(n)
            update()

def launch():
    global launched
    if len(inputs)==4 and not launched:
        launchtext['text']="Status: Launched w/ inputs "+str(inputs) 
        launched=True
        update()

def resetf():
    att=[]
    launched=False
    executed=False
    training=False
    trained=False
    testing=False
    accdisplayed=False
    queried=False
    inputs=[0,1]
    atts['text']=''
    atts['fg']="black"
    initext['text']=''
    initext2['text']=''
    initext3['text']=''
    initext4['text']=''
    initext5['text']=''
    initext6['text']=''
    initext7['text']=''
    launchtext['text']="Status: Not Launched"
    update()

#All buttons in GUI
a1=Button(root,image=p1, command= lambda: c1("Reported Chest Pain Type",1), height=60, width=100)
a1.pack()
a1.place(x=800,y=200)

a2=Button(root,image=p2, command= lambda: c1("Resting Blood Pressure",2), height=60, width=100)
a2.pack()
a2.place(x=800,y=290)

a3=Button(root,image=p3, command= lambda: c1("Serum Cholesterol",3),height=60, width=100)
a3.pack()
a3.place(x=800,y=380)

a4=Button(root,image=p4, command= lambda: c1("Fasting Blood Sugar Exceeds 120?",4),height=60, width=100)
a4.pack()
a4.place(x=800,y=470)

a5=Button(root,image=p5, command= lambda: c1("Resting Electrocardiograph Results",5),height=60, width=100)
a5.pack()
a5.place(x=800,y=560)

a6=Button(root,image=p6, command= lambda: c1("Maximum Heart Rate Achieved",6),height=60, width=100)
a6.pack()
a6.place(x=950,y=200)

a7=Button(root,image=p7, command= lambda: c1("Exercise-Induced Angina?",7),height=60, width=100)
a7.pack()
a7.place(x=950,y=290)

a8=Button(root,image=p8, command= lambda: c1("ST Depression Induced by Exercise\nRelative to Rest",8),height=60, width=100)
a8.pack()
a8.place(x=950,y=380)

a9=Button(root,image=p9, command= lambda: c1("Slope of Peak Exercise ST Segment",9),height=60, width=100)
a9.pack()
a9.place(x=950,y=470)

a10=Button(root,image=p10, command= lambda: c1("Number of Major Vessels Colored\nby Cardiovascular Angiography Testing",10),height=60, width=100)
a10.pack()
a10.place(x=950,y=560)

launch=Button(root, bg="#932d0e",text="Launch",command=launch,font=("Garamond",40),height=1, width=8)
launch.pack()
launch.place(x=1225,y=200)

d1=Button(height=100,width=1,bg="#a1dbdf")
d1.pack()
d1.place(x=1133,y=150)

d2=Button(height=100,width=1,bg="#a1dbdf")
d2.pack()
d2.place(x=700,y=150)


resett=Button(root,image=reset,command=resetf,height=70, width=112)
resett.pack()
resett.place(x=1200,y=760)

relaunchh=Button(root,image=relaunch,height=70, width=112)
relaunchh.pack()
relaunchh.place(x=1350,y=760)

#Gets the loop running
root.after(1000,launchloop)

root.mainloop()
