# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:59:45 2020

@author: lenovo
"""


import GoogleMapDownloader 
from GoogleMapDownloader  import downloadImage
from model_test import generateResult
from tkinter import *
from tkinter.ttk import * 
from tkinter import filedialog
from PIL import ImageTk, Image
from model_test2 import generateResults
def ClearImage():
    print("succes")
    canvas.delete("all")
def GenerateResult():
    print("success")
    generateResult()
    messagebox.showinfo("Message","Processing done Successfully!")
def GenerateResults():
    print("success")
    generateResults()
    messagebox.showinfo("Message","Processing done Successfully!")
    
def viewImage():
    print("succes")
    win3 = Tk()
    win3.geometry("800x800")
    win3.title("Satellite Image")
    canvas4 = Canvas(win3, width = 800, height = 800)  
    canvas4.grid( row=1,column = 1,sticky='EWNS' ,padx = 5,pady=5) 
    load=Image.open("high_resolution_image.jpg")
    load = load.resize((800, 800), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(load,master=win3)
    canvas4.create_image( 0,0,anchor=NW, image=img)
    win3.mainloop()
def viewPie():
    print("succes")
    win2 = Tk()
    win2.geometry("600x600")
    win2.title("PieChart Analysis")
    canvas3 = Canvas(win2, width = 600, height = 600)  
    canvas3.grid( row=1,column = 1,sticky='EWNS' ,padx = 5,pady=5) 
    load=Image.open("piechart.png")
    load = load.resize((600, 600), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(load,master=win2)
    canvas3.create_image( 0,0,anchor=NW, image=img)
    win2.mainloop()
def viewBar():
    print("succes")
    win1 = Tk()
    win1.geometry("900x600")
    win1.title("BarGraph Analysis")
    canvas2 = Canvas(win1, width = 900, height = 600)  
    canvas2.grid( row=1,column = 1,sticky='EWNS' ,padx = 5,pady=5) 
    load=Image.open("bargraph.png")
    load = load.resize((900, 600), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(load,master=win1)
    canvas2.create_image( 0,0,anchor=NW, image=img)
    win1.mainloop()
def DownloadImage():
    print("Street: "+e1.get()+" City: "+e2.get()+"Country: "+e3.get())
    downloadImage(e1.get(),e2.get(),e3.get())
    messagebox.showinfo("Message","Satellite Image Downloaded Successfully!")
    load=Image.open("high_resolution_image.jpg")
    load = load.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(load)
    button4=Button(root,text='Proceed', command=GenerateResults)
    canvas.create_window(150,275,window=button4)
    #canvas.create_image( 0,0,anchor=NW, image=img)
    root.mainloop()
    
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    load=Image.open(filename)
    load = load.resize((300, 300), Image.ANTIALIAS)
    load.save("high_resolution_image.jpg")
    img = ImageTk.PhotoImage(load)
    button0=Button(root,text='Proceed', command=GenerateResult)
    canvas.create_window(150,275,window=button0)
    button01=Button(root,text='Clear', command=ClearImage)
    canvas.create_window(150,300,window=button01)
    canvas.create_image( 0,0,anchor=NW, image=img)
    #canvas.delete(id1)
    root.mainloop()

def DownloadAction(event=None):
    Label(root, text="Enter Street Name").grid(column=3,row=1,pady=5)
    Label(root, text="Enter City").grid(column=3,row=2,pady=5)
    Label(root, text="Enter Country").grid(column=3,row=3,pady=5)
    e1.grid(row=1, column=4,pady=5)
    e2.grid(row=2, column=4,pady=5)
    e3.grid(row=3, column=4,pady=5)
    button5 = Button(root, text='Download Image', command=DownloadImage)
    button5.grid(row = 4, column = 4,padx=5, pady=5) 
    root.mainloop()
    
    
root = Tk()
root.geometry("700x400")
root.title("Main Window")
#style = Style() 
#style.configure('TButton', font = ('calibri', 20, 'bold'), borderwidth = '4',foreground = 'green', background = 'black') 

#style.map('TButton', foreground = [('active', '! disabled', 'green')], background = [('active', 'black')]) 

button1 = Button(root, text='Upload Image',command=UploadAction)
button1.grid(row = 1, column = 1,columnspan=1,sticky='EWNS',padx = 5,pady=5)#, padx = 200, pady=5) 
#button1.pack()
button2 = Button(root, text='Download Satellite Image', command=DownloadAction)
button2.grid(row = 2, column = 1,sticky='EWNS',padx = 5,pady=5)#, padx = 200, pady=5) 
#button2.pack()
button3 = Button(root, text = 'Quit !', command = root.destroy) 
#button3.pack()
button3.grid(row = 3, column = 1,sticky='EWNS',padx = 5,pady=5)#, padx = 200,pady=5) 
button4 = Button(root, text = 'View Zoom Image', command = viewImage) 
#button3.pack()
button4.grid(row = 1, column = 5,sticky='EWNS',padx = 5,pady=5)#, padx = 200,pady=5) 
button5 = Button(root, text = 'View Bar Graph Analysis', command = viewBar) 
#button3.pack()
button5.grid(row = 2, column = 5,sticky='EWNS',padx = 5,pady=5)#, padx = 200,pady=5) 
button6 = Button(root, text = 'View Pie Chart Analysis', command = viewPie) 
#button3.pack()
button6.grid(row = 3, column = 5,sticky='EWNS',padx = 5,pady=5)#, padx = 200,pady=5) 
canvas = Canvas(root, width = 300, height = 300)  
canvas.grid( row=1,column = 3,rowspan=4,columnspan=2,sticky='EWNS' ,padx = 5,pady=5)  
e1 = Entry(root)
e2 = Entry(root)
e3 = Entry(root)
root.mainloop()

