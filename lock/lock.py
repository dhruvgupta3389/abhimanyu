
import sys
sys.path.insert(1, 'D://jarves_backend//speak')
import speak

import random
def notallow():
    a=random.randint(1,5)
    if a==1:
        speak.speak("I can't do that")
    if a==2:
        speak.speak("I do not have that permission")
    if a==3:
        speak.speak("i am not able to do that")
    if a==4:
        speak.speak("i do not to acces")
    if a==5:
        speak.speak("i do not have that permission")
def check(id):
    if id==1:
        speak.speak("welcome master dhruv")
    if id==2:
        speak.speak("we have spical message for you want hear that ")
        a=input("yes or no")
        if a.lower()=="yes":
            speak.speak("i love u gungun i know i already told many time but what will i do i never any idea that u love me or this is fustrating just by this i ask u do u love me or not")
            
        elif a.lower()=="no":
            pass
        else:
            speak.speak("not in option")
