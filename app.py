from flask import Flask, request, render_template
from PIL import Image
import io
from utils import extract_face, predict


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        
        img = Image.open(request.files['file1'])
        data = io.BytesIO()
        img = extract_face(img)
        img.save(data, "JPEG")
        pred = predict(img)
        return render_template("uploaded.html", pred=pred)
    return render_template("index.html")

app.run(host='0.0.0.0', port=5000, threaded=True)