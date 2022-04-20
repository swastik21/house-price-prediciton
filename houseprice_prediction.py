# -*- coding: utf-8 -*-

from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import *
import math
from PIL import ImageTk,Image

win = Tk()
top = Toplevel()
win.geometry('400x400')
win.resizable(0,0)
image1= PhotoImage(file = "image//house.png")
imglabel= Label(win, image=image1)
imglabel.place(x=0,y=0,relwidth=1,relheight=1)
win.title('House Price Prediction')
top.withdraw()
streeteasy = pd.read_csv("data//manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms','size_sqft','min_to_subway']]

y = df[['price']]

arr1 = np.array(df['price'])
normalized_rent = preprocessing.normalize([arr1])

arr2 = np.array(df['min_to_subway'])
normalized_subway = preprocessing.normalize([arr2])

plt.scatter(df['size_sqft'],df['price'])
plt.xlabel("Size in sqft")
plt.ylabel("Price")
plt.show()

plt.scatter(normalized_subway,normalized_rent)
plt.xlabel("distance to subway in mins")
plt.ylabel("Price")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)
y_predict = pd.DataFrame(y_predict)
print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

y_test = y_test.head(10)
y_test = np.array(y_test)
y_predict = y_predict.head(10)
y_predict = np.array(y_predict)

plt.plot(y_test,y_predict,'o')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

print(lm.coef_,lm.intercept_)




Label(win,text="House Price Prediction",font=("Aerial",20,"bold"),fg="maroon").pack()

size_var = StringVar()
subway_var = StringVar()
bedroom_var = StringVar()
predict_var = StringVar()

Label(win,text="Size in Sqft",font="Aerial 15 bold",fg='red').place(x=140,y=70)
size_entry = Entry(win,textvariable=size_var,fg='red',justify='center').place(x=140,y=100)
Label(win,text="Minutes to Subway",font="Aerial 15 bold",fg='red').place(x=110,y=140)
subway_entry = Entry(win,textvariable=subway_var,fg='red',justify='center').place(x=140,y=170)
Label(win,text="Number of Bedrooms",font="Aerial 15 bold",fg='red').place(x=100,y=210)
subway_entry = Entry(win,textvariable=bedroom_var,fg='red',justify='center').place(x=140,y=240)

def predict():
    predictvar = 0 
    size = size_var.get()
    subway = subway_var.get()
    bedroom = bedroom_var.get()
    size = int(size)
    bedroom = int(bedroom)
    subway = int(subway)
    apartment = [[bedroom,size,subway]]
    print(apartment)
    predictvar = model.predict(apartment)
    print("Predicted price: $%.2f" % predictvar)
    predict_var.set("$"+str(math.floor(predictvar[0][0])))
    
    

Button(win,text="Predict",font="Aerial 15 bold",bg="white",fg="green",padx=2,command=predict).place(x=160,y=270)
Label(win,text="Predicted Price",font="Aerial 15 bold",fg='green').place(x=130,y=330)
Entry(win,textvariable=predict_var,fg='green',justify='center').place(x=140,y=370)

win.mainloop()