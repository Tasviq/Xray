from flask import Flask, render_template, request, redirect, url_for
from model import load_pneumonia_model, predict_pneumonia
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

load_pneumonia_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        prediction = predict_pneumonia(filepath)
        if prediction[0][0] > 0.5:
            result_text = "Result is Normal"
        else:
            result_text = "Pneumonia Detected"
        
        return render_template('result.html', prediction=result_text, image_file=file.filename)
    
        os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)
