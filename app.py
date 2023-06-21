from flask import Flask, render_template
import flask
from PIL import Image
from predict import predict
import numpy as np

app = Flask(__name__, template_folder='frontend/template', static_folder='frontend/static')

@app.route('/')
def home():
    print('Render index.html')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_frontend():

    imagefile = flask.request.files.get('imagefile', '')
    image = Image.open(imagefile).convert('RGB')
    image = image.resize((180, 180), Image.ANTIALIAS)
    data = np.asarray(image)

    predicted_class = predict(data)
    return render_template('index.html', prediction_text=predicted_class)


if __name__=='__main__':
    app.run(debug=True)