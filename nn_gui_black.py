from tkinter import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K 
import numpy as np
import os
import time
clock=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
score=0
x=[]
prediction=0
y=[]

# load cleveland heart disease dataset
dataset = np.loadtxt("cleveland_hd.csv", delimiter=",")

#names of the attributes
names=["Age (years)","Sex (1=male, 0=female)", "Reported chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)","Resting blood pressure (mm Hg)","Serum cholesterol (mg/dL)","Does fasting blood sugar exceed 120 mg/dL threshold? (1=yes, 0=no)","Resting electrocardiographic testing results (0=normal, 1=having ST-T wave abnormality, 2=showing probable or definite left ventricular hypertrophy)","Maximum heart rate achieved (bpm)","Exercise-induced angina? (1=yes, 0=no)","ST depression induced by exercise relative to rest","Slope of the peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)","Number of major vessels (0-3) colored by flourosopy"]

#creates the Neural Network
def createNN(X, Y):      
    global score   
    K.clear_session()
    model = Sequential()

    #layers: 4 -> 1000 -> 50 -> 1
    model.add(Dense(1000, input_dim=4, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #training loop for Neural Network
    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=150, validation_split=0.2, batch_size=10, verbose=2)

    #tests Neural Network
    scores = model.evaluate(X, Y)

    #if the score is an error value, redo the trial.
    if (scores[1]>0.5385 and scores[1]<0.5390) or (scores[1]>0.4611 and scores[1]<0.4613): 
        if time.clock()-clock>35:
            score=101
            return 0
        else:
            print("    "+"Error occured. Rerunning trial.")
            return 1
    else:
        score=(scores[1]*100)
        return 0

def createNNP(X, Y, Z):      
    global score
    global prediction   
    K.clear_session()
    model = Sequential()

    #layers: 4 -> 1000 -> 50 -> 1
    model.add(Dense(1000, input_dim=4, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #training loop for Neural Network
    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=150, validation_split=0.2, batch_size=10, verbose=2)

    #tests Neural Network
    scores = model.evaluate(X, Y)

    # #if the score is an error value, redo the trial.
    # if (scores[1]>0.5385 and scores[1]<0.5390) or (scores[1]>0.4611 and scores[1]<0.4613): 
    #     if time.clock()-clock>35:
    #         score=101
    #         return 0
    #     else:
    #         print("    "+"Error occured. Rerunning trial.")
    #         return 1
    # else:
    #     score=(scores[1]*100)
    #     return 0

    prediction=(model.predict(Z))

#trial
def trial(X,Y):

    #this loop facilitates the rerunning of trials
    succ=createNN(X, Y)
    if succ==1:
        trial(X,Y)

#trial
def trialp(X,Y,Z):

    #this loop facilitates the rerunning of trials
    succ=createNNP(X, Y, Z)
    if succ==1:
        trialp(X,Y,Z)

#creating the GUI Window
root = Tk()

#Makes the window fullscreen
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.focus_set()

#Pressing escape exits the program
root.bind("<Escape>", lambda e: root.quit())

#Title at the top of GUI
titlel = Label(root, text="Heart Disease Neural Network Simulation", fg="#d30404", bg="black", font=("Helvetica", 40)) 
titlel.pack()

#Our names
names = Label(root, text='''Mani Chadaga, Akshay Nambudripad, and Alexander "Sasha" Hydrie. Ninth Graders at Central High School''' , fg="#840505", bg="black", font=("Helvetica", 24)) 
names.pack()
#Set background color
root.configure(background="black")

#Initalize the ~~ten~~ TWELVE photos for the button
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
atttoptext=Label(root, font=("Helvetica", 24),fg="#8e241c", bg="black",text="Attributes:                ")
atttoptext.pack()
atttoptext.place(x=800,y=650)

step1=Label(root, font=("Helvetica", 18), bg="black",text="1) Choose 2 Attributes:", fg='red')
step1.pack()
step1.place(x=800,y=150)

step2=Label(root, font=("Helvetica", 18), bg="black",text="2) Launch Neural Network", fg='red')
step2.pack()
step2.place(x=1200,y=150)

launchtext=Label(root, font=("Helvetica", 14), fg="#8e241c", bg="black",text="Status: Not Launched",width=37)
launchtext.pack()
launchtext.place(x=1140,y=330)

initext=Label(root, font=("Helvetica", 14), fg="#f44e42",bg="black",text="",width=37)
initext.pack()
initext.place(x=1140,y=380)

initext2=Label(root, font=("Helvetica", 14), fg="#f44e42", bg="black",text="",width=37)
initext2.pack()
initext2.place(x=1140,y=430)

initext4=Label(root, font=("Helvetica", 14), fg="#f44e42", bg="black",text="",width=37)
initext4.pack()
initext4.place(x=1140,y=530)

initext5=Label(root, font=("Helvetica", 24), fg="#0dc4b7", bg="black",text="",width=21)
initext5.pack()
initext5.place(x=1140,y=580)

initext6=Label(root, font=("Helvetica", 24), fg="#1df2e4", bg="black",text="",width=21)
initext6.pack()
initext6.place(x=1140,y=630)

initext7=Label(root, font=("Helvetica", 14), fg="#f44e42", bg="black",text="",width=37)
initext7.pack()
initext7.place(x=1140,y=690)

initext3=Label(root, font=("Helvetica", 14), fg="#f44e42", bg="black",text="",width=37)
initext3.pack()
initext3.place(x=1140,y=480)

atts=Label(root,fg="#8e241c", font=("Helvetica", 11), bg="black",text="",width=28, anchor='w', justify='left')
atts.pack()
atts.place(x=800,y=683)

warntext=Label(root, font=("Helvetica", 14), bg="black",text="",fg="#932d0e")
warntext.pack()
warntext.place(x=755,y=780)

ltext1=Label(root, font=("Helvetica", 24), fg="#d32424", bg="black",text="What is a Neural Network?")
ltext1.pack()
ltext1.place(x=18,y=130)
ltext2=Label(root,  anchor='w', justify='left',font=("Helvetica", 14), fg="#d35454", bg="black",text='''A neural network is a type of computer program that imitates how our brains\ndeduce information based on multiple different factors. It generally takes in a\nvariety of inputs and comes to one specific output based on these ideas. At\nfirst, the neural network may be completely incorrect, but it refines its\ndeduction algorithm to the point where its accuracy is acceptable. It is\nable to improve accuracy through multivariable calculus and linear algebra.''')
ltext2.pack()
ltext2.place(x=25,y=170)

ltext3=Label(root, font=("Helvetica", 24), fg="#d32424", bg="black",text="What does our NN accomplish?")
ltext3.pack()
ltext3.place(x=18,y=330)
ltext4=Label(root,  anchor='w', justify='left',font=("Helvetica", 14), fg="#d35454", bg="black",text='''Our neural network’s inputs are all factors linked to heart disease, such as\nage, sex, chest pain type, blood sugar, and ECG. We found pre existing\ndata of these attributes for 297 patients, and the dataset also mentioned\nwhether each patient had heart disease or not. Using this data, our neural\nnetwork trained itself to be able to predict a diagnosis of heart disease.\nOur project came to life when we refined our NN to only accept 2 attributes\n(in addition to age and sex, which were used every time) and train/test\nbased on those attributes. This allowed us to analyze the accuracies and\ndetermine which of the attributes were most indicative of heart disease.''')
ltext4.pack()
ltext4.place(x=25,y=370)

ltext11=Label(root, font=("Helvetica", 21), fg="#239599", bg="black",text="Risk Calculator: Enter Your Own Values")
ltext11.pack()
ltext11.place(x=18,y=645)


# E1=Entry(root)
# E1.pack()
# E1.place(x=25,y=685)
# ltext12=Label(root,  anchor='w', justify='left',font=("Helvetica", 14), fg="#34dde2", bg="black",text='''First, press the "Reset Program Entirely" button (bottom right corner). Then,\nselect the two (2) attributes which the Neural Network will train based upon.\nIf you mistakenly select an attribute, repress its button. When you are happy\nwith the attributes, press "Launch." Allow up to 40 seconds for the program to\ncomplete. If the NN is successfully able to train, then you will be given an\naccuracy representing the proportion of heart disease patients that the NN\ncorrectly diagnosed.''')
# ltext12.pack()
# ltext12.place(x=25,y=685)

choices=['Reported Chest\n Pain', 'Resting Blood\n Pressure', 'Serum Cholesterol', 'Fasting Blood\n Pressure Exceeds 120?','Resting ECG\n Test Results','Maximum Heart\n Rate Achieved','Exercise-Induced\n Angina?','ST Depression Induced\n by Exercise Relative\n to Rest','Slope of Peak Exercise\n ST Segment','# of Major Vessels\n Colored by Cardio-\nvascular Angiography']

dicto={}
for i in range(10):
    dicto[choices[i]]=i+1

choice_var=StringVar()
choice_var.set("Select your\n1st Attribute\nfor this NN")
opt_menu = OptionMenu (root, choice_var, *choices)
opt_menu.config(width=18)
opt_menu.pack()
opt_menu.place(x=25, y=690)
inputsm=[]
choice_var2=StringVar()
choice_var2.set("Select your\n2nd Attribute\nfor this NN")
opt_menu2 = OptionMenu (root, choice_var2, *choices)
opt_menu2.config(width=18)
opt_menu2.pack()
opt_menu2.place(x=25, y=760)
att1v=0
att2v=0
agev=0
sexv=0

blah=StringVar(root)

att1q=OptionMenu(root, blah, 'h')
att1l=Label()
att1qe=Entry()

att2q=OptionMenu(root, blah, 'h')
att2l=Label()
att2qe=Entry()

def set1p():
    git=choice_var.get()

    global att1q
    global att1l
    global att1qe

    if git=='Reported Chest\n Pain':
        var=StringVar(root)

        options=['Typical\n Angina','Atypical\n Angina','Non-Anginal\n Pain','Asymp-\ntomatic']
        dicte={}
        for i in range(4):
            dicte[options[i]]=i+1

        att1q=OptionMenu(root, var, 'Typical\n Angina','Atypical\n Angina','Non-Anginal\n Pain','Asymp-\ntomatic',command=(lambda event, e=var: sav1(dicte[e.get()]))) 
        att1q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Select your\ntype of\nChest Pain')
        att1q.pack()
        att1q.place(x=230, y=735)

    if git=='Resting Blood\n Pressure':
        var=StringVar()
        att1qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att1qe.bind('<FocusOut>', (lambda event, e=att1qe: sav1(e.get())))  
        att1l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Resting\nBlood Pressure\n(integer 90-210)',font=("Helvetica", 10))
        att1l.pack()
        att1l.place(x=230,y=760)   
        att1qe.pack()
        att1qe.place(x=265, y=735) 

    if git=='Serum Cholesterol':
        var=StringVar()
        att1qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att1qe.bind('<FocusOut>', (lambda event, e=att1qe: sav1(e.get())))  
        att1l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Serum\nCholesterol\n(integer 120-570)',font=("Helvetica", 10))
        att1l.pack()
        att1l.place(x=230,y=760)   
        att1qe.pack()
        att1qe.place(x=265, y=735) 

    if git=='Fasting Blood\n Pressure Exceeds 120?':
        var=StringVar(root)

        options=['No','Yes']
        dicte={}
        for i in range(2):
            dicte[options[i]]=i

        att1q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav1(dicte[e.get()]))) 
        att1q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Does Fasting\nBlood Sugar\nExceed 120?')
        att1q.pack()
        att1q.place(x=230, y=735)    

    if git=='Resting ECG\n Test Results':
        var=StringVar(root)

        options=['Normal','Having ST-T\n wave\n abnormality','Probable or\n definite left\n ventricular hypertrophy']
        dicte={}
        for i in range(3):
            dicte[options[i]]=i

        att1q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav1(dicte[e.get()]))) 
        att1q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Select your\nECG results.')
        att1q.pack()
        att1q.place(x=230, y=735)   

    if git=='Maximum Heart\n Rate Achieved':
        var=StringVar()
        att1qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att1qe.bind('<FocusOut>', (lambda event, e=att1qe: sav1(e.get())))  
        att1l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Maximum\nHeart Rate\n(integer 65-205)',font=("Helvetica", 10))
        att1l.pack()
        att1l.place(x=230,y=760)   
        att1qe.pack()
        att1qe.place(x=265, y=735)  

    if git=='Exercise-Induced\n Angina?':
        var=StringVar(root)

        options=['No','Yes']
        dicte={}
        for i in range(2):
            dicte[options[i]]=i

        att1q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav1(dicte[e.get()]))) 
        att1q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Does exercise\ninduce\nangina? (Y/N)')
        att1q.pack()
        att1q.place(x=230, y=735)


    if git=='ST Depression Induced\n by Exercise Relative\n to Rest':
        var=StringVar()
        att1qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att1qe.bind('<FocusOut>', (lambda event, e=att1qe: sav1(e.get())))  
        att1l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Enter exercise\ninduced ST\nDepression wrt rest\n(integer 0-7)',font=("Helvetica", 10))
        att1l.pack()
        att1l.place(x=230,y=760)   
        att1qe.pack()
        att1qe.place(x=265, y=735)  

    if git=='Slope of Peak Exercise\n ST Segment':
        var=StringVar(root)

        options=['Upsloping','Flat', 'Downsloping']
        dicte={}
        for i in range(3):
            dicte[options[i]]=i+1

        att1q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav1(dicte[e.get()]))) 
        att1q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('What is the\nST segment\nslope? Pick one.')
        att1q.pack()
        att1q.place(x=230, y=735)

    if git=='# of Major Vessels\n Colored by Cardio-\nvascular Angiography':
        var=StringVar()
        att1qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att1qe.bind('<FocusOut>', (lambda event, e=att1qe: sav1(e.get())))  
        att1l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Enter number of\nmajor vessels\ncolored by C.A.\n(integer 0-3)',font=("Helvetica", 10))
        att1l.pack()
        att1l.place(x=230,y=760)   
        att1qe.pack()
        att1qe.place(x=265, y=735)    

def set2p():

    global att2q
    global att2l
    global att2qe

    git=choice_var2.get()
    if git=='Reported Chest\n Pain':
        var=StringVar(root)

        options=['Typical\n Angina','Atypical\n Angina','Non-Anginal\n Pain','Asymp-\ntomatic']
        dicte={}
        for i in range(4):
            dicte[options[i]]=i+1

        att2q=OptionMenu(root, var, 'Typical\n Angina','Atypical\n Angina','Non-Anginal\n Pain','Asymp-\ntomatic',command=(lambda event, e=var: sav2(dicte[e.get()]))) 
        att2q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Select your\ntype of\nChest Pain')
        att2q.pack()
        att2q.place(x=380, y=735)

    if git=='Resting Blood\n Pressure':
        var=StringVar()
        att2qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att2qe.bind('<FocusOut>', (lambda event, e=att2qe: sav2(e.get())))  
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Resting\nBlood Pressure\n(integer 90-210)',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=380,y=760)   
        att2qe.pack()
        att2qe.place(x=415, y=735) 

    if git=='Serum Cholesterol':
        var=StringVar()
        att2qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att2qe.bind('<FocusOut>', (lambda event, e=att2q: sav2(e.get())))  
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Serum\nCholesterol\n(integer 120-570)',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=380,y=760)   
        att2qe.pack()
        att2qe.place(x=415, y=735) 

    if git=='Fasting Blood\n Pressure Exceeds 120?':
        var=StringVar(root)

        options=['No','Yes']
        dicte={}
        for i in range(2):
            dicte[options[i]]=i

        att2q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav2(dicte[e.get()]))) 
        att2q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Does Fasting\nBlood Sugar\nExceed 120?')
        att2q.pack()
        att2q.place(x=380, y=735)    

    if git=='Resting ECG\n Test Results':
        var=StringVar(root)

        options=['Normal','Having ST-T\n wave\n abnormality','Probable or\n definite left\n ventricular hypertrophy']
        dicte={}
        for i in range(3):
            dicte[options[i]]=i

        att2q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav2(dicte[e.get()]))) 
        att2q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Select your\nECG results.')
        att2q.pack()
        att2q.place(x=380, y=735)   

    if git=='Maximum Heart\n Rate Achieved':
        var=StringVar()
        att2qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att2qe.bind('<FocusOut>', (lambda event, e=att2qe: sav2(e.get())))  
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Click above and\nenter your Maximum\nHeart Rate\n(integer 65-205)',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=380,y=760)   
        att2qe.pack()
        att2qe.place(x=415, y=735)  

    if git=='Exercise-Induced\n Angina?':
        var=StringVar(root)

        options=['No','Yes']
        dicte={}
        for i in range(2):
            dicte[options[i]]=i

        att2q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav2(dicte[e.get()]))) 
        att2q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('Does exercise\ninduce\nangina? (Y/N)')
        att2q.pack()
        att2q.place(x=380, y=735)


    if git=='ST Depression Induced\n by Exercise Relative\n to Rest':
        var=StringVar()
        att2qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att2qe.bind('<FocusOut>', (lambda event, e=att2qe: sav2(e.get())))  
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Enter exercise\ninduced ST\nDepression wrt rest\n(integer 0-7)',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=380,y=760)   
        att2qe.pack()
        att2qe.place(x=415, y=735)  

    if git=='Slope of Peak Exercise\n ST Segment':
        var=StringVar(root)

        options=['Upsloping','Flat', 'Downsloping']
        dicte={}
        for i in range(3):
            dicte[options[i]]=i+1

        att2q=OptionMenu(root, var, *options,command=(lambda event, e=var: sav2(dicte[e.get()]))) 
        att2q.config(width=10,height=3, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
        var.set('What is the\nST segment\nslope? Pick one.')
        att2q.pack()
        att2q.place(x=380, y=735)

    if git=='# of Major Vessels\n Colored by Cardio-\nvascular Angiography':
        var=StringVar()
        att2qe=Entry(root, textvariable=var, width=5, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
        att2qe.bind('<FocusOut>', (lambda event, e=att2qe: sav2(e.get())))  
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Enter number of\nmajor vessels\ncolored by C.A.\n(integer 0-3)',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=380,y=760)   
        att2qe.pack()
        att2qe.place(x=415, y=735)

def nothing(att2q):
    print('hi')

def sav1(a):
    global att1v
    att1v=int(a)
    print(att1v)

def sav2(b):
    global att2v
    att2v=int(b)
    print(att2v)

def savage(c):
    global agev
    agev=int(c)
    print(agev)

def savsex(d):
    global sexv
    sexv=int(d)
    print(sexv)

def evaluate(event):
    print(att2q.get())

def setup():
    if choice_var.get()=="Select your\n1st Attribute\nfor this NN" or choice_var2.get()=="Select your\n2nd Attribute\nfor this NN":
        tf2['text']="You must select two (2) attributes to use for this NN."
    att1=dicto[choice_var.get()]
    att2=dicto[choice_var2.get()]
    if att1==att2:
        tf2['text']="You have chosen the same attribute twice. Please retry."
    else:
        submit=Button(root,bg="green",text="Click white box (top-left) then\nclick here to launch NN",command=submitg,width=23)
        submit.pack()
        submit.place(x=530, y=690)

        global inputsm
        tf3=Label(root,font=("Helvetica", 7), fg="#f44e42", bg="black",text="",width=18)
        tf3.pack()
        tf3.place(x=230, y=690)
        tf4=Label(root,font=("Helvetica", 7), fg="#f44e42", bg="black",text="",width=18)
        tf4.pack()
        tf4.place(x=380, y=690)
        tf2['text']="Setup Complete. You may now enter your attribute values."
        tf['command']=nothing
        tf3['text']=choice_var.get()
        set1p()
        tf4['text']=choice_var2.get()
        set2p()
        inputsm=[0,1,att1+1,att2+1]
          
        att2l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Age:',font=("Helvetica", 10))
        att2l.pack()
        att2l.place(x=530,y=660)   
        att23l=Label(root, fg='#d32424',bg='black',anchor='w', justify='left',text='Sex:',font=("Helvetica", 10))
        att23l.pack()
        att23l.place(x=595,y=660) 
        ageq.pack()
        ageq.place(x=565, y=660)
        satt2q.pack()
        satt2q.place(x=630, y=655)

tf=Button(root,bg="#932d0e",text="Set Up",command=setup,width=20)
tf.pack()
tf.place(x=26, y=830)

def resetv(e):
    global att1v
    global att2v
    # global att1q
    # global att1l
    # global att1qe    
    # global att2q
    # global att2l
    # global att2qe
    tf['command']=setup
    # att1q.destroy()
    # att1l.destroy()
    # att1qe.destroy()
    # att2q.destroy()
    # att2l.destroy()
    # att2qe.destroy()
    bb=Label(root, bg='black',text='',width=72,height=9)
    bb.pack()
    bb.place(x=200,y=690)

    choice_var.set("Select your\n1st Attribute\nfor this NN")
    choice_var2.set("Select your\n2nd Attribute\nfor this NN")

    d4=Button(height=8,width=2,bg="black")
    d4.pack()
    d4.place(x=200,y=690)

    d4=Button(height=8,width=2,bg="black")
    d4.pack()
    d4.place(x=350,y=690)

    d4=Button(height=8,width=2,bg="black")
    d4.pack()
    d4.place(x=500,y=690)

    att1v=0
    att2v=0



 

def submitg():
    Z=np.array(((agev,sexv,att1v,att2v),(0,1,69,1)))*1.0
    print(Z)
    print(inputsm)
    X = dataset[:,inputsm]
    Y = dataset[:,13]
    for k in range(len(Y)):
        if Y[k]!=0:
            Y[k]=1  
    print(X)
    trialp(X,Y,Z)
    print(prediction[0])
    print(score)
    submitt=Label(root,bg="black",font=("Helvetica", 16),fg="black",text="You have",width=15)
    submitt.pack()
    submitt.place(x=520, y=730)

    submitt2=Label(root,bg="black",font=("Helvetica", 16),fg="black",text="of heart disease.",width=15)
    submitt2.pack()
    submitt2.place(x=525, y=790)
    submitt['fg']='white'
    submitt2['fg']='white'
    risk=Label(root,bg="black",font=("Helvetica", 20),fg="black",text="No Risk",width=11)
    risk.pack()
    risk.place(x=525, y=755)
    if prediction[0]<0.2:
        risk['fg']='#99ff33'
        risk['text']="No Risk"
    elif prediction[0]<0.4:
        risk['fg']='#ccff33'
        risk['text']="Slight Risk"
    elif prediction[0]<0.6:
        risk['fg']='#ffff00'
        risk['text']="Moderate Risk"
    elif prediction[0]<0.8:
        risk['fg']='#ff9900'
        risk['text']="High Risk"
    else:
        risk['fg']='#cc3300'
        risk['text']="Extreme Risk"

    tf2['text']="Click this text to reset, or change your values & relaunch."
    tf2.bind("<Button-1>",resetv)

ph=Entry(root,width=1)
ph.pack()
ph.place(x=0,y=0)


agevar=StringVar()
ageq=Entry(root, textvariable=agevar, width=3, fg='#d32424', bg='black',highlightbackground="red",font=("Helvetica", 10))
ageq.bind('<FocusOut>', (lambda event, e=ageq: savage(e.get())))
tf2=Label(root,font=("Helvetica", 14), anchor='w', justify='left',fg="#f44e42", bg="black",text="",width=47)
tf2.pack()
tf2.place(x=200, y=830)

tf3=Label(root,font=("Helvetica", 7), fg="#f44e42", bg="black",text="",width=18)
tf3.pack()
tf3.place(x=230, y=690)
tf4=Label(root,font=("Helvetica", 7), fg="#f44e42", bg="black",text="",width=18)
tf4.pack()
tf4.place(x=380, y=690)


d4=Button(height=8,width=2,bg="black")
d4.pack()
d4.place(x=200,y=690)

d4=Button(height=8,width=2,bg="black")
d4.pack()
d4.place(x=350,y=690)

d4=Button(height=8,width=2,bg="black")
d4.pack()
d4.place(x=500,y=690)

svar=StringVar()

soptions=['Female','Male']
sdicte={}
for i in range(2):
    sdicte[soptions[i]]=i

satt2q=OptionMenu(root, svar, *soptions,command=(lambda event, e=svar: savsex(sdicte[e.get()]))) 
satt2q.config(width=4,height=1, fg='#d32424',bg='black',activebackground = "black",activeforeground='red',highlightbackground="red",font=("Helvetica", 10))  
svar.set('Pick')



# E2=Entry(root)
# E2.pack()
# E2.place(x=125,y=825)
# # ltext12=Label(root,  anchor='w', justify='left',font=("Helvetica", 14), fg="#34dde2", bg="black",text='''First, press the "Reset Program Entirely" button (bottom right corner). Then,\nselect the two (2) attributes which the Neural Network will train based upon.\nIf you mistakenly select an attribute, repress its button. When you are happy\nwith the attributes, press "Launch." Allow up to 40 seconds for the program to\ncomplete. If the NN is successfully able to train, then you will be given an\naccuracy representing the proportion of heart disease patients that the NN\ncorrectly diagnosed.''')
# # ltext12.pack()
# # ltext12.place(x=25,y=685)
# choice_var2=StringVar()
# opt_menu2 = OptionMenu (root, choice_var2, '1', '2', '3', '4','5','6','7','8','9','10')
# opt_menu2.pack()
# opt_menu2.place(x=225, y=825)
# ltext112=Label(root, font=("Helvetica", 22), fg="red", bg="black",text="You have heart disease")`
# ltext112.pack()
# ltext112.place(x=250,y=750)

# ltext12=Label(root,  anchor='w', justify='left',font=("Helvetica", 10), fg="#34dde2", bg="black",text='''TouchScreen: Enabled''')
# ltext12.pack()
# ltext12.place(x=1400,y=0)

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
        if score==101:
            initext3['fg']="red"
            initext3['text']='Timeout occured. NN was likely unable\nto train itself using the specified\nattributes. Relaunch, or reset and try\na completely different pair of attributes.\n\n\n: ^('
            trained=True
            testing=True
            accdisplayed=True
            queried=True
        else:
            initext3['fg']="#f44e42"
            initext3['text']='NN Self-Training Complete.'
            trained=True
    if launched and executed and not training:
        time.sleep(1) 
        initext3['text']='NN Self-Training Commenced...'
        training=True
    if launched and not executed:
        global clock
        clock=time.clock()
        initext['text']='Dataset Initialized'
        X = dataset[:,inputs]
        print(inputs)
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
        atts['fg']='#a8554f'
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
    global inputs
    if len(inputs)==4 and not launched:
        launchtext['text']="Status: Launched w/ desired inputs"
        launched=True
        update()

def resetf():
    global inputs
    global att
    global launched
    global executed
    global score
    global training
    global trained
    global testing
    global accdisplayed
    global queried
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
    atts['fg']="#8e241c"
    initext['text']=''
    initext2['text']=''
    initext3['text']=''
    initext4['text']=''
    initext5['text']=''
    initext6['text']=''
    initext7['text']=''
    launchtext['text']="Status: Not Launched"
    update()

def relaunchf():
    global inputs
    global att
    global launched
    global executed
    global score
    global training
    global trained
    global testing
    global accdisplayed
    global queried
    launched=False
    executed=False
    training=False
    trained=False
    testing=False
    accdisplayed=False
    queried=False
    # K.clear_session()
    if len(inputs)==4 and not launched:
        initext['text']=''
        initext2['text']=''
        initext3['text']=''
        initext4['text']=''
        initext5['text']=''
        initext6['text']=''
        initext7['text']=''
        launchtext['text']="Status: Relaunched w/ desired inputs"
        update()   
        launched=True

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

launch=Button(root, bg="#932d0e",text="Launch",command=launch,font=("Helvetica",40),height=1, width=8)
launch.pack()
launch.place(x=1225,y=200)

d1=Button(height=100,width=2,bg="black")
d1.pack()
d1.place(x=1133,y=150)

d2=Button(height=100,width=2,bg="black")
d2.pack()
d2.place(x=710,y=150)

d3=Button(height=1,width=100,bg="black")
d3.pack()
d3.place(x=0,y=600)


resett=Button(root,image=reset,command=resetf,height=70, width=112)
resett.pack()
resett.place(x=1200,y=760)

relaunchh=Button(root,command=relaunchf,image=relaunch,height=70, width=112)
relaunchh.pack()
relaunchh.place(x=1350,y=760)

#Gets the loop running
root.after(1000,launchloop)

root.mainloop()
