from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route("/clasification", methods=["POST"])
def clasification():
    if request.method == "POST":
        if request.files.get("text"):
            user_review = request.files["text"].read()
            
            model = load_model("app\modelo4.h5")
            maxlen = 256
            
            word_index = imdb.get_word_index()
            user_tokens = [
                word_index[word] if word in word_index else 0
                for word in user_review.split()
            ]

            user_padded = pad_sequences([user_tokens], maxlen=maxlen)

            prediction = model.predict(user_padded)[0][0]
            if prediction >= 0.5:
                print(f"Prediction: {prediction}")
                print("Positive review!")
                mensaje = "Positive review!"
                return mensaje
            else:
                print(f"Prediction: {prediction}")
                print("Negative review.")
                mensaje = "Negative review."
                return mensaje
        
        else:
            return "No text file provided."

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0')
