import base64
import io

import numpy as np
from flask import Flask, jsonify, redirect, request
from PIL import Image
from skimage.transform import resize  # <--- resize
from tensorflow.keras.models import load_model

app = Flask(__name__)

main_html = """
<html>
<head></head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

   function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
   }

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");


      symbols_name = ["heart","diamond","club","spade"]
      symbols = ["♥", "♦", "♣", "♠"];
      mensaje_symbols = symbols.join(",")      
      document.getElementById('mensaje').innerHTML  = 'Dibuja uno de estos simbolos ' + mensaje_symbols;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
  	    $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
  }



</script>
<body onload="InitThis();">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>
    <div align="left">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="300"/>
    </div>
    <div align="center">
        <h1 id="mensaje">Dibujando...</h1>
        <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
        <br/>
        <br/>
        <button onclick="javascript:clearArea();return false;">Borrar</button>
    </div>
    <div align="center">
      <form method="post" action="predict" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Predecir">
      </form>
    </div>
</body>
</html>
"""


@app.route("/")
def main():
    return main_html


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_data = request.form.get("myImage").replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))

        # Redimensionar la imagen a 28x28
        img = img.resize((28, 28))

        # Convertir la imagen a numpy array y normalizar los valores de píxeles
        img = np.array(img) / 255.0

        # Agregar una dimensión para el lote y el canal
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.expand_dims(img[:, :, 0], axis=-1)
        # Cargar el modelo y realizar la predicción
        model = load_model("modelo.h5")
        prediction = model.predict(img)

        # Obtener la clase predicha
        predicted_class = int(np.argmax(prediction))

        # Devolver la predicción como una respuesta JSON
        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        print("Error occurred:", e)
        return redirect("/", code=302)


if __name__ == "__main__":
    app.run()