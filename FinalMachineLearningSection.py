#Imports all here:
import tkinter as tk #for UX design of the code
from tkinter import filedialog
from tkinter import font
from tkinter import ttk

#Code to Upload Image
def uploadFunction():
        root.withdraw()
        file_paths = filedialog.askopenfilenames(title="Please upload the food that you have! Align them against a solid background beside eachother.")
        for file_path in file_paths: #filename will be the picture that we are dealing with
            print(f'User uploaded image! (For testing: filename is {file_path}\n)', file_path) 


#The following is not able to run on google colab because it is ux design
#The UX Design for after back-end is made
root=tk.Tk() #make the tk object
root.title("DS3000 Machine Learning Project") #the title
root.geometry("500x500") #width by height of the app

#Homescreen for the app
label = tk.Label(root, text = "Welcome to Remake - The food upcycling app.", font=('Times', 20))
label.pack(pady=5)
label2 = tk.Label(root, text="What will we make today? (Upload your food below)", font = ('Times', 14))
label2.pack(pady=5)
label5 = tk.Label(root, text="Please upload the food that you have! Align them against a solid background beside eachother.", font = ('Times', 14))
label5.pack(pady=5)
button = ttk.Button(root, text='Upload', command=uploadFunction)
button.pack(pady=5)
    
    
label3 = tk.Label(root, text="Lets see what we can make with that...\n", font=('Times', 14))
label3.pack(pady=5)
label4= tk.Label(root, text="Thank you so much for using Remake today!", font = ('Times', 14))
#=======================HERE INSERT MACHINE LEARNING MODEL===========================
#recognition of the food.
#print like (our model recognized foods x, y, z)
#call a recipe with those foods in our dataset or say like no recipe found, but recipe fund for food x"
#print recipe



root.mainloop()




