
from flask import Flask, request, render_template
import pickle
import re
from gemini_translate import translate_response
app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("symptom_list.pkl", "rb") as f:
    symptoms = pickle.load(f)
def extract_symptoms(user_input):
    input_lower = user_input.lower()
    return [sym for sym in symptoms if re.search(r'\b' + re.escape(sym) + r'\b', input_lower)]
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    translation = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        target_lang = request.form["language"]
        found = extract_symptoms(user_input)
    if not found:
      response = "Sorry, I couldn't identify any known symptoms. Please try again."
    else:
       input_vector = [1 if sym in found else 0 for sym in symptoms]
       prediction = model.predict([input_vector])[0]
       response = f"Based on your symptoms, the predicted disease is: {prediction}"

    translation = translate_response(response, target_lang)

    return render_template("index.html", response=response, translation=translation)
