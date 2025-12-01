#Imports all here:
import tkinter as tk #for UX design of the code
from tkinter import filedialog
from PIL import Image, ImageTk #for background
from tkinter import ttk
import os #for terminal trouble shooting
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from PIL import Image
import pandas as pd



#=======HUGE NOTE ====== 
# In Tkinter, which is the UI design library I am using, any time you use anything in it, you must pack it. Whether it be a label or canvas or image. Other wise It will not display
#print("Script File:", __file__) - this was some troubleshooting I did to try and find where my image file was
#print("\nWorking Directory:", os.getcwd())
#print("Files actually in working directory", os.listdir())

global userImage;

#Code for user to Upload Image
def uploadFunction(): #function that occurs to upload file
        global userImage;
        global matching_recipes;
        root.withdraw() #hides the root
        file_paths = filedialog.askopenfilenames(title="Please upload the food that you have! Align them against a solid background beside eachother.") #message for uploading food
        for file_path in file_paths: #upload image
            print(f'User uploaded image! (For testing: filename is {file_path}\n)', file_path) #print to terminal about the uploaded image
            userImage=Image.open(file_path)
        root.deiconify()#return to window 
        text3=canvas.create_text(800, 400, text="Lets see what we can make with that...", font="times 20", fill="grey") #lets user know image has been uploaded
        backgroundBox3=canvas.bbox(text3) #text sizing for box around text
        rect3=canvas.create_rectangle(backgroundBox3, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect3,text3) #text location

        #======================================================================
        # Create a tuple of names for the indexes of prediction so we get a useful output
        #======================================================================
        train_dir = "fruits-and-vegetables/versions/2/dataset/train"
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        #print(f"Number of classes: {len(class_names)}") #for troubleshooting only
        #print(f"Classes: {class_names}") #for troubleshooting only

        #Bring In our Model
        model_path = "fruitandveggieclassifier.pth"

        #Load In The Rest of Pretrained ResNet
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        state_dict = torch.load(model_path, map_location="cpu") #load model
        num_classes = state_dict['fc.weight'].shape[0] #the number of classes in the model
        #below loads in the neural network
        model.fc = nn.Linear(model.fc.in_features, num_classes) #the fully connected layer
        model.load_state_dict(state_dict) #load it!
        model.eval()
        print("Model loaded!")

        #Image Preprocessing for resnet
        transform = transforms.Compose([
            transforms.Resize((224, 224)), #resize photo to 224 by 224 because thats what resnet will take
            transforms.ToTensor(), #make it a tensor
            transforms.Normalize([0.485, 0.456, 0.406], #nromalize he data
                                [0.229, 0.224, 0.225])
        ])

        #open the image
        img = userImage
        img2 = transform(img).unsqueeze(0) #torch needs a batch dimension input for the model to understand your image, so the unsqueezing does that, and the transform applies the above transformations to our image

        #ACTUAL MODEL PREDICTION BELOW
        with torch.no_grad(): #so basically telling the model to not compute gradients because we arent training the model anymore
            output = model(img2)#sends model to our neural network for prediction
            proba = torch.softmax(output, dim=1)
        
        threshold = 0.02
        pred_indices = (proba >= threshold).nonzero(as_tuple=True)[1]  # indices above threshold
        if len(pred_indices) == 0:
             pred_indices = torch.topk(proba, 1, dim=1).indices[0]

        pred_probs = [proba[0, i].item() for i in pred_indices]
        pred_classes = [class_names[i] for i in pred_indices]
        sorted_preds = sorted(zip(pred_classes, pred_probs), key=lambda x: x[1], reverse=True)

        #print(f"Prediction index: {pred.item()}") #for testing only
        text_output = "Foods Detected:\n" + "\n".join([f"{cls}" for cls, prob in sorted_preds])
        text4 = canvas.create_text(800, 500, text=text_output, font="times 20", fill="grey")
        backgroundBox4=canvas.bbox(text4) #text sizing for box around text
        rect4=canvas.create_rectangle(backgroundBox4, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect4,text4) #text location
       

        text5 = canvas.create_text(800, 600, text="Retrieving Recipes with the given food...", font="times 20", fill="grey")
        backgroundBox5=canvas.bbox(text5) #text sizing for box around text
        rect5=canvas.create_rectangle(backgroundBox5, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect5,text5) #text location
        matching_recipes=[]
        recipes=pd.read_csv("archive/ingredient.csv")
      
        for pred in pred_classes:
            matches = recipes[recipes['Ingredients'].str.contains(pred, case=False, na=False)]
            for idx, row in matches.iterrows():
                matching_recipes.append({
                    "Recipe": row['Title'],
                    "Ingredients": row['Ingredients'],
                    "Instructions": row['Instructions']
                })
        matching_recipes = [dict(t) for t in {tuple(d.items()) for d in matching_recipes}]
        if matching_recipes:
            window=tk.Toplevel()
            canvasNewWindow=tk.Canvas(window)
            scrollbar=tk.Scrollbar(window, orient="vertical", command=canvasNewWindow.yview)
            canvasNewWindow.configure(yscrollcommand=scrollbar.set)
            canvasNewWindow.pack(side="left",fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            scroll_frame=tk.Frame(canvasNewWindow)
            canvasNewWindow.create_window((0,0), window=scroll_frame, anchor="nw")
            title_label = tk.Label(scroll_frame, text="Matching Recipes Below:", font=("Times", 30, "bold"), fg="grey", bg="White", borderwidth=2, relief="ridge")
            title_label.pack(pady=5)
            recipe_text=""
            for recipe in matching_recipes:
                block = (
                    f"======={recipe['Recipe']}=======\n"
                    f"Ingredients:\n{recipe['Ingredients']}\n\n"
                    f"Instructions:\n{recipe['Instructions']}\n"
                    "----------------------------------------\n"
                )

                lbl = tk.Label(
                    scroll_frame,
                    text=block,
                    font=("Times", 16,),
                    justify="left",
                    wraplength=1500,
                    anchor="w",
                    bg="white",
                    fg="grey"
                    
                )
                lbl.pack(pady=10, anchor="w")


            scroll_frame.update_idletasks()
            canvasNewWindow.configure(scrollregion=canvasNewWindow.bbox("all"))
        else:
            recipe_text = "No matching recipes found for your ingredients."
            text5 = canvas.create_text(800, 600, text=recipe_text, font="times 18", fill="grey", anchor="n", width=900)
            backgroundBox5 = canvas.bbox(text5)  # get the bounding box for rectangle
            rect5 = canvas.create_rectangle(backgroundBox5, fill='white', outline='grey')
            canvas.tag_lower(rect5, text5)
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




