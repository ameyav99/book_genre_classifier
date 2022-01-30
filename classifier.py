from flask import Flask, request, jsonify
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)



# Load neural network when Flask boots up
model = load_model("lstm_v1.h5")

with open('tokenizer.pickle', 'rb') as handle:
    global tokenizer 
    tokenizer = pickle.load(handle)

@app.route('/api/classify', methods=['POST'])
def classify():
    content = request.json
    errors = []
    title = request.form.get('title')

    if len(title)<1 :
        errors.append("Missing Title")
    if len(errors)<1:
        random_input=tokenizer.texts_to_sequences([title])
        padded_sequence = pad_sequences(random_input, maxlen=70, padding='post', truncating='post')
        out=model.predict(padded_sequence)
        class_id=np.argmax(out)
        dict = {0:'Digital electronics and logic design', 1:'Data Structures',2:'Computer Networks',3:'DBMS',4:'Cyber Security',5:'Java',6:'c++',7:'python',8:'Web Development',9:'Computer Vision',10:'Data Analysis',11:'Natural Language Processing',12:'Internet of Things',13:'Computer Organization and Architecture',14:'Distributed System',15:'Operating System'}
        response= {"class_id":str(class_id),"class":dict[class_id]}
    else:
        response = {"errors":errors}
   #print(content)   
    return jsonify(response)

if __name__ == '__main__':
    app.run()