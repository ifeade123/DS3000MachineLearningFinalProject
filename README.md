**=<3=About Remake!=<3=**

Remake is a project made by 4 university students, tasked with using AI to solve a real world issue.
Our Program Works to help people find recipes for food in their fridges quickly and easily when they upload a photo to the app in the code. 
The goal is to hopefully help people reduce food waste in the case that they do not know what to cook with food in their fridges.



**=<3=How To Use=<3=**

Steps:
1. Simply run the 'FinalMachineLearningSection.py'code, and the app window should pop up. (Warning, do not run the foodidentifiermodel.py, or else you will retrain the whole model, its up to you, but I wouldnt. 🤷🏾‍♀️)
2. Upload a '.jpg' image to the app, and wait. The app will open a new window with retrieved recipes.
3. Scroll, cook, and enjoy!



**=<3=About The Model Performance Metrics=<3=**

We used a ResNet pre-trained model, and trained the last layer using a kaggle dataset on fruits and veggies: https://www.kaggle.com/datasets/abhisheksubhashswami/fruits-and-vegetables.
We used 10 epochs, and then made the rest of the app using regular python, as well as libraries such as tkinter, PIL, torch, and pandas.

<img width="2001" height="1990" alt="ConfusionMatrixDs3000" src="https://github.com/user-attachments/assets/10f6a5a8-f5a0-4c0d-bfe0-e7bf139e4198" />

Our confusion matrix having the trend through the diagonal of the graph represents instances where the model recognizes a class correctly. Any lightly purple coloured spots of the matrix that are not on that diagonal spot are places where the model is predicting incorrectly. Based on the diagonal part being green, and most of the exterior to that being purple, we can see that the model mostly predicts class correctly.

<img width="567" height="432" alt="DS3000Precision Recall" src="https://github.com/user-attachments/assets/49b229df-4ac1-4e9d-86ed-7408ad198e1e" />

The Precision v.s. Recall Graph having points that tend to be on the top right part of the graph shows that our model generally has a good balance between Precision and Recall. 

<img width="551" height="427" alt="F1Score" src="https://github.com/user-attachments/assets/86cdd2fa-7675-4840-adb5-e0761fe34cbb" />

Our F1 Score Bar Chart represents the balance between precision and recall for each class and the higher scores indicate that the model is doing a good job balancing between correctly identifying True Positives and avoiding False Positives and False Negatives. 

Thank you for reading this, and have fun with Remake! <3<3<3<3
