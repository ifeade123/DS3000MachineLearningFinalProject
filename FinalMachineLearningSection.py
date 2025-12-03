
#*********IMPORTS BELOW*********
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

#=======Some troubleshooting======== 
# print("Script File:", __file__) - this was some troubleshooting I did to try and find where my image file was
#print("\nWorking Directory:", os.getcwd())
#print("Files actually in working directory", os.listdir())

#========Variable Declaration========
global userImage; #for image uploading

#=================<3=================Function for uploading image, and everything after that=================<3=================

def uploadFunction(): #function that occurs to upload file
        
        #++++++++++Variable Declaration+++++++++++++++
        global userImage; #for later image use
        global matching_recipes; #for finding matching imahes
       
        #++++++++++++Uploading Photo From Files+++++++++++++++++++
        root.withdraw() #hides the original window
        file_paths = filedialog.askopenfilenames(title="Please upload the food that you have! Align them against a solid background beside eachother.") #message for uploading food
        for file_path in file_paths: #upload image
            print(f'User uploaded image! (For testing: filename is {file_path}\n)', file_path) #print to terminal about the uploaded image
            userImage=Image.open(file_path) #user image is the file that the user upliaded
        root.deiconify()#return to window 

        #++++++++++++++++++Message On Main Window For User+++++++++++++++++++++++++++
        text3=canvas.create_text(800, 400, text="Lets see what we can make with that...", font="times 20", fill="grey") #lets user know image has been uploaded
        backgroundBox3=canvas.bbox(text3) #text sizing for box around text
        rect3=canvas.create_rectangle(backgroundBox3, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect3,text3) #text location

        #++++++++++++++++++Load Up Model Info and its paths+++++++++++++++++++++++++++++
        train_dir = "fruits-and-vegetables/versions/2/dataset/train" #the directory to the training set
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]) #sorts the fruits and vegetable paths
        #print(f"Number of classes: {len(class_names)}") #for troubleshooting only
        #print(f"Classes: {class_names}") #for troubleshooting only
        model_path = "fruitandveggieclassifier.pth" #insert the model we trained
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) #Load In The Rest of Pretrained ResNet
        state_dict = torch.load(model_path, map_location="cpu") #load model on cpu
        num_classes = state_dict['fc.weight'].shape[0] #the number of classes in the model
        
        #++++++++++++++Load In The Actual Fully Connected Neural Network+++++++++++++++
        model.fc = nn.Linear(model.fc.in_features, num_classes) #the fully connected layer
        model.load_state_dict(state_dict) #load it!
        model.eval() #model evaluation mode to turn off certain badly behaving features
        print("Model loaded!") #notify terminal that model is loaded

        #+++++++++++++++++++++Image Preprocessing for resnet+++++++++++++++++++++++
        transform = transforms.Compose([
            transforms.Resize((224, 224)), #resize photo to 224 by 224 because thats what resnet will take
            transforms.ToTensor(), #make it a tensor
            transforms.Normalize([0.485, 0.456, 0.406], #nromalize he data
                                [0.229, 0.224, 0.225])])

        #+++++++++++++++++++Open image and  transform it so it can be used in model++++++++++++++++++++++++++
        img = userImage
        img2 = transform(img).unsqueeze(0) #torch needs a batch dimension input for the model to understand your image, so the unsqueezing does that, and the transform applies the above transformations to our image

        #++++++++++++++++++++++++++++Actually Have The Model Predict+++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad(): #so basically telling the model to not compute gradients because we arent training the model anymore
            output = model(img2)#sends model to our neural network for prediction
            proba = torch.softmax(output, dim=1)
        threshold = 0.02 #this is the threshold that the model needs to consider the food prediction as a valid prediction. It is very low so that it can predict multiple foods at a time
        pred_indices = (proba >= threshold).nonzero(as_tuple=True)[1]  # indices above threshold
        if len(pred_indices) == 0: #if the number of predicted foods is 0(either because of th threshold or otherwise), 
             pred_indices = torch.topk(proba, 1, dim=1).indices[0] #then it will just select the highest value predicted in the model

        #the following assigns the probability and predictions from the model to a value and puts it in a tuplle
        pred_probs = [proba[0, i].item() for i in pred_indices]
        pred_classes = [class_names[i] for i in pred_indices]
        sorted_preds = sorted(zip(pred_classes, pred_probs), key=lambda x: x[1], reverse=True) #this sorts the predictions

        #print(f"Prediction index: {pred.item()}") #for testing only

        #++++++++++++++++++++++++++++Have model state what it detected++++++++++++++++++++++++++++++++++++++
        text_output = "Foods Detected:\n" + "\n".join([f"{cls}: {100*prob:.2f}% Certainty" for cls, prob in sorted_preds]) #this is output that will go back to the user to tell them what the model found
        text4 = canvas.create_text(800, 500, text=text_output, font="times 20", fill="grey") #this is UI design to show the output of what the model found
        backgroundBox4=canvas.bbox(text4) #text sizing for box around text
        rect4=canvas.create_rectangle(backgroundBox4, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect4,text4) #text location
       
        #++++++++++++++++++++++++++++++Let user know we are retrieving recipes++++++++++++++++++++++++++++++++++++++
        text5 = canvas.create_text(800, 600, text="Retrieving Recipes with the given food...", font="times 20", fill="grey") #lets user know we are looking for the recipes matching the food
        backgroundBox5=canvas.bbox(text5) #text sizing for box around text
        rect5=canvas.create_rectangle(backgroundBox5, fill='white', outline='grey') #creating rectangle for box around text
        canvas.tag_lower(rect5,text5) #text location
        matching_recipes=[] #we create a list of recipes, blank at the moment but it will hold all the recipes that match the food 
        recipes=pd.read_csv("archive/ingredient.csv") #recipes are read from the CSV
      
        #+++++++++++++++++++++++++++++++Make a data frame of matching recipes+++++++++++++++++++++++++++++++++++++++++++++
        for pred in pred_classes: #for each prediction
            matches = recipes[recipes['Ingredients'].str.contains(pred, case=False, na=False)] #make sure that in the ingredients, they have a string that matches the name of our prediction, irrespective of case, and no missing values
            for idx, row in matches.iterrows():  #this whole thing just adds recipe, ingredients, and instructuons of matching ingredients to the list
                matching_recipes.append({
                    "Recipe": row['Title'],
                    "Ingredients": row['Ingredients'],
                    "Instructions": row['Instructions']
                })
        matching_recipes = [dict(t) for t in {tuple(d.items()) for d in matching_recipes}] 

        #+++++++++++++++++++++++++++++++++++++If there are matching recipes, make a new window and show them++++++++++++++++++++++++++++++++++++++
        if matching_recipes: #if we have matching recipes
            window=tk.Toplevel() #create a new window
            backgroundimagerecipes = tk.PhotoImage(file="RBg.png") #load in the background picture for the app
            
            canvasNewWindow=tk.Canvas(window) #and a new blank canvas
            canvasNewWindow.pack(side="left",fill="both", expand=True) #pack the canvas

            canvasNewWindow.create_image(0,0, image=backgroundimagerecipes, anchor="nw") #anchor and place the image in the window
            canvasNewWindow.bg_image=backgroundimagerecipes #assign th background photo to a value for now

            scrollbar=tk.Scrollbar(window, orient="vertical", command=canvasNewWindow.yview) #make a scroll bar so user can scroll through recipes
            canvasNewWindow.configure(yscrollcommand=scrollbar.set) #configure the canvas with scrollbar
            scrollbar.pack(side="right", fill="y") #pack the scrollbar, but keep it on the right
            
            scroll_frame=tk.Frame(canvasNewWindow, bg="white", bd=0) #mak the frame that we scroll wuith
            canvasNewWindow.create_window((220,0), window=scroll_frame, anchor="nw") #create the canvas that th labels will lie on, center it by moving it 220.
            
            title_label = tk.Label(scroll_frame, text="Matching Recipes Below:", font=("Times", 30, "bold"), fg="grey", bg="White", borderwidth=2, relief="ridge",justify="center") #this is text, showing the user the recipes below
            title_label.pack(pady=5) #pack the title label
            recipe_text="" #initialize recipe text as an empty string
            for recipe in matching_recipes: # for every recipe that matches, print the recipe, ingredients, and instructions
                block = (f"======={recipe['Recipe']}=======\n"
                    f"Ingredients:\n{recipe['Ingredients']}\n\n"
                    f"Instructions:\n{recipe['Instructions']}\n"
                    "----------------------------------------\n")
                
                lbl = tk.Label(scroll_frame, text=block, font=("Times", 16,),justify="center",wraplength=1100,anchor="w",bg="white",fg="grey") #the below is the label that holds all the recipe text, and its formatting
                lbl.pack(pady=10) #pack the label
            
            #+++++++++++++++++++++++++Have scroll update as you move through prints+++++++++++++++++++++++++++++++++++++++++++++++++++
            canvasNewWindow.update_idletasks() #update the window as we move
            scroll_height2 = len(matching_recipes)*len(block)/60 #set the height of the scroll
            canvas_width = canvasNewWindow.winfo_width() #store the width of the canvas
            bg_height = canvasNewWindow.bg_image.height() #store the background image height

            y = 0 #only y value for tiling
            while y < scroll_height2: #while y is less than the scroll height
                canvasNewWindow.create_image(0, y, image=canvasNewWindow.bg_image, anchor="nw") #create a new image in the y for the background after each previous one
                y += bg_height #add the heights
            canvasNewWindow.configure(scrollregion=(0, 0, canvas_width, scroll_height2)) #set the scroll region on the canvas

        #+++++++++++++++++++++++++++++++++++++Otherwise, if no matching recipes+++++++++++++++++++++++++++++++++++++++
        else: #otherwise
            recipe_text = "No matching recipes found for your ingredients." #tell user no matching recipes
            text5 = canvas.create_text(800, 600, text=recipe_text, font="times 18", fill="grey", anchor="n", width=900) #text formatting
            backgroundBox5 = canvas.bbox(text5)  # get the bounding box for rectangle
            rect5 = canvas.create_rectangle(backgroundBox5, fill='white', outline='grey') #text formatting
            canvas.tag_lower(rect5, text5) #test fromatting

#(((((((((((((((((((((((((((((UI CREATION FOR USER)))))))))))))))))))))))))))))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Initialization for UI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
root=tk.Tk() #make the tk object #creates a root for the whole UI design to be based on, this is essentially the app
root.title("DS3000 Machine Learning Project") #the title of app
root.geometry("1850x1000") #width by height of the app, when it initially is oppened

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Make a background image for UI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
canvas=tk.Canvas(root, width=2000, height=1500) #canvas is like drawing on the app without covering anything. We make the size if the canvas large
canvas.pack(pady=5) #pack tha canvas, pad it in the y direction by 5, meaning that nothing can touch it above or below in the y direction by 5
backgroundimage = tk.PhotoImage(file="RemakeBackground.png") #load in the background picture for the app
canvas.create_image(0,0, image=backgroundimage, anchor="nw") #anchor and place the image in the window

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%welcome message for app and formatting%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text1=canvas.create_text(800, 100, text="Welcome to Remake - The food upcycling app!", font="times 40 bold", fill="grey") 
backgroundBox1=canvas.bbox(text1) #text formatting
rect1=canvas.create_rectangle(backgroundBox1, fill='white', outline='grey')
canvas.tag_lower(rect1,text1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%secondary message for app, prompting user to upload image and formatting%%%%%%%%%%%%%%%%%%%%%%%%
text2=canvas.create_text(800, 200, text="What will we make today?\nPlease upload the food that you have(as .jpg)! Align them against a solid background beside eachother.", font="times 20", fill="grey")
backgroundBox2=canvas.bbox(text2) #text formatting
rect2=canvas.create_rectangle(backgroundBox2, fill='white', outline='grey') #textformatting
canvas.tag_lower(rect2,text2) #text formatting

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%button for uploading image, using Tkinter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
button = ttk.Button(root, text='Upload', command=uploadFunction) #For users to upload image
button.pack(pady=0) #pack image
buttonWindow=canvas.create_window(800, 300, window=button) #create the button on canvas

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Run main loop%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
root.mainloop() #call to be able to run the tkinter code