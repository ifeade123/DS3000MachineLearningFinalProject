#Imports all here:
import tkinter as tk #for UX design of the code
from tkinter import filedialog
from PIL import Image, ImageTk #for background
from tkinter import ttk
import os #for terminal trouble shooting


#=======HUGE NOTE ====== 
# In Tkinter, which is the UI design library I am using, any time you use anything in it, you must pack it. Whether it be a label or canvas or image. Other wise It will not display
#print("Script File:", __file__) - this was some troubleshooting I did to try and find where my image file was
#print("\nWorking Directory:", os.getcwd())
#print("Files actually in working directory", os.listdir())


#Code for user to Upload Image
def uploadFunction(): #function that occurs to upload file
        root.withdraw() #hides the root
        file_paths = filedialog.askopenfilenames(title="Please upload the food that you have! Align them against a solid background beside eachother.") #message for uploading food
        for file_path in file_paths: #upload image
            print(f'User uploaded image! (For testing: filename is {file_path}\n)', file_path) #print to terminal about the uploaded image
        root.deiconify()#return to window 
        text3=canvas.create_text(800, 400, text="Lets see what we can make with that...", font="times 20", fill="grey") #lets user know image has been uploaded
        backgroundBox3=canvas.bbox(text3) #text sizing for box around text
        rect3=canvas.create_rectangle(backgroundBox3, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect3,text3) #text location

root=tk.Tk() #make the tk object #creates a root for the whole UI design to be based on, this is essentially the app
root.title("DS3000 Machine Learning Project") #the title of app
root.geometry("1850x1000") #width by height of the app, when it initially is oppened

canvas=tk.Canvas(root, width=2000, height=1500) #canvas is like drawing on the app without covering anything. We make the size if the canvas large
canvas.pack(pady=5) #pack tha canvas, pad it in the y direction by 5, meaning that nothing can touch it above or below in the y direction by 5
backgroundimage = tk.PhotoImage(file="RemakeBackground.png") #load in the background picture for the app
canvas.create_image(0,0, image=backgroundimage, anchor="nw") #anchor and place the image in the window

#welcome message for app and formatting
text1=canvas.create_text(800, 100, text="Welcome to Remake - The food upcycling app!", font="times 40 bold", fill="grey") 
backgroundBox1=canvas.bbox(text1) #text formatting
rect1=canvas.create_rectangle(backgroundBox1, fill='white', outline='grey')
canvas.tag_lower(rect1,text1)

#secondary message for app, prompting user to upload image and formatting
text2=canvas.create_text(800, 200, text="What will we make today?\nPlease upload the food that you have! Align them against a solid background beside eachother.", font="times 20", fill="grey")
backgroundBox2=canvas.bbox(text2)
rect2=canvas.create_rectangle(backgroundBox2, fill='white', outline='grey')
canvas.tag_lower(rect2,text2)

#button for uploading image, using Tkinter
button = ttk.Button(root, text='Upload', command=uploadFunction)
button.pack(pady=0)
buttonWindow=canvas.create_window(800, 300, window=button)

#=======================HERE INSERT MACHINE LEARNING MODEL===========================
#recognition of the food.
#print like (our model recognized foods x, y, z)
#call a recipe with those foods in our dataset or say like no recipe found, but recipe fund for food x"
#print recipe

#closing message for app, and formatting
'''
text4=canvas.create_text(800, 400, text="\nThank you so much for using Remake today!", font="times 20", fill="grey") #to be inserted after our machine learning model is
backgroundBox4=canvas.bbox(text4)
rect4=canvas.create_rectangle(backgroundBox4, fill='white', outline='grey')
canvas.tag_lower(rect4,text4)'''

root.mainloop()




