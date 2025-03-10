from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
with open("mail_detection.pkl", 'rb') as file:
    model = pickle.load(file)

with open("vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)