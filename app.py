import os
from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import tensorflow as tf
from tensorflow import keras



app = Flask(__name__)
UPLOAD_FOLDER = "/Users/KUSHAL GADIGE/Desktop/Project1/static"




@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, model)
            print(pred)
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":

    model = keras.models.load_model("models/mobilenetv2_model.h5")


    def predict(image_path, model):
        img=keras.preprocessing.image.load_img(image_path, target_size=(150,150))
        x = keras.preprocessing.image.img_to_array(img)
        x=np.expand_dims(x, axis=0)

        predictions = model.predict(x)
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        if predictions.numpy() == 0:
            return "Its a Cat"
        else:
            return "Its a Dog"
        

    app.run(port=12000, debug=True)