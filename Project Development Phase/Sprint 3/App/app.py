from flask import Flask,render_template,request
from tensorflow.keras.preprocessing import image
import os
import numpy as np #used for numerical analysis
from keras.models import load_model#to load our trained model
import requests


app = Flask(__name__,template_folder="templates") # initializing a flask app
# Loading the model
model=load_model('nutrition.h5')
print("Loaded model from disk")


@app.route('/')# route to display the home page
def home():
    return render_template("home.html")#rendering the home page

@app.route('/image1',methods=['GET','POST'])# routes to the index html
def image1():
    return render_template("/image.html")



@app.route('/predict',methods=['GET', 'POST'])# route to show the predictions in a web UI
def launch():
    if request.method=='POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to an array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image

        pred=np.argmax(model.predict(x), axis=1)
        print("prediction",pred)#printing the prediction
        index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']
        
        result=str(index[pred[0]])
                    
        x=result
        print(x)
        result=nutrition(result)
        print(result)
        
        return render_template("0.html",showcase=(result),showcase1=(x))
def nutrition(index):


    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"
    
    querystring = {"query":index}
    
    headers = {
        'x-rapidapi-key': "261814b785msh8d97a1ef3a76170p113cfajsn73a68a80b147",
        'x-rapidapi-host': "calorieninjas.p.rapidapi.com"
        }
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    print(response.text)     
    return response.json()['items']
if __name__ == "__main__":
   # running the app
    app.run(debug=False)
