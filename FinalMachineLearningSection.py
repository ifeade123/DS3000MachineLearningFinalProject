#Imports all here:
from google.colab import files #for user ability to upload images
import tkinter as tk #for UX design of the code

#Code to Upload Image
while(True):
  print("Welcome to Remake - The food upcycling app! \n")
  print("Please upload the food that you have! Align them against a solid background beside eachother.\n")
  uploadedfile = files.upload() #for the file that the user uploads
  for filename, file in uploadedfile.items(): #filename will be the picture that we are dealing with
    print(f'User uploaded image! (For testing: filename is {filename}\n)', filename)
    print("Lets see what we can make with that...\n")

    #=======================HERE INSERT MACHINE LEARNING MODEL===========================
    #recognition of the food.
    #print like (our model recognized foods x, y, z)
    #call a recipe with those foods in our dataset or say like no recipe found, but recipe fund for food x"
    #print recipe

    print("Thank you so much for using Remake today!\n")


#The following is not able to run on google colab because it is ux design
#The UX Design for after back-end is made
root=tk.Tk() #make the tk object
root.title("DS3000 Machine Learning Project") #the title
root.geometry("500x500") #width by height of the app

#Homescreen for the app
label = tk.Label(root, text = "Welcome to Remake - The food upcycling app.")
label.pack(pady=150)
label2 = tk.Label(root, text="What will we make today?")
label2.pack(pady=250)

root.mainloop()

