from flask import Flask, jsonify, request, make_response
import jwt
from functools import wraps
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow 
import random
import json
import pickle
import redis
import config



class Chatbot():
    def __init__(self, category):

        redisUser = config.cred()
        self.redisClient = redis.Redis(host=redisUser['host'], port=redisUser['port'], password= redisUser['password'])

        self.stemmer = LancasterStemmer()
        self.category = category

        with open(f"./Category{self.category}/ques.json", encoding="utf-8") as file:
            self.data = json.load(file)
        
        try:
            with open(f"Category{self.category}/data.pickle","rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)

        except:
            self.words = []
            self.labels = []
            self.docs_x = []
            self.docs_y = []

            for intent in self.data["intents"]:
                for pattern in intent["pattern"]:
                    wrds = nltk.word_tokenize(pattern)
                    self.words.extend(wrds)
                    self.docs_x.append(wrds)
                    self.docs_y.append(intent["tag"])

                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

            self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in "?"]
            self.words = sorted(list(set(self.words)))

            self.labels = sorted(self.labels)

            self.training = []
            self.output = []

            self.out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(self.docs_x):
                bag = []

                wrds = [self.stemmer.stem(w) for w in doc]

                for w in self.words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)
                
                self.output_row = self.out_empty[:]
                self.output_row[self.labels.index(self.docs_y[x])] = 1

                self.training.append(bag)
                self.output.append(self.output_row)


            self.training = numpy.array(self.training)
            self.output = numpy.array(self.output)

            with open(f"Category{self.category}/data.pkl","wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

        tensorflow.reset_default_graph()

        # Change number of nodes in hidden layers
        if self.category == 1:
            net = tflearn.input_data(shape=[None, len(self.training[0])])
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, len(self.output[0]),activation="softmax")
            net = tflearn.regression(net)

            self.model = tflearn.DNN(net)

            # self.model.fit(self.training, self.output, n_epoch=500, batch_size=8, show_metric=True)
            # self.model.save(f"./Category{self.category}/chatbot.tflearn")

        elif self.category == 2:
            net = tflearn.input_data(shape=[None, len(self.training[0])])
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, len(self.output[0]),activation="softmax")
            net = tflearn.regression(net)

            self.model = tflearn.DNN(net)

            # self.model.fit(self.training, self.output, n_epoch=500, batch_size=8, show_metric=True)
            # self.model.save(f"./Category{self.category}/chatbot.tflearn")

        elif self.category == 3:
            net = tflearn.input_data(shape=[None, len(self.training[0])])
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, len(self.output[0]),activation="softmax")
            net = tflearn.regression(net)

            self.model = tflearn.DNN(net)

            # self.model.fit(self.training, self.output, n_epoch=850, batch_size=8, show_metric=True)
            # self.model.save(f"./Category{self.category}/chatbot.tflearn")


        elif self.category == 4:
            net = tflearn.input_data(shape=[None, len(self.training[0])])
            net = tflearn.fully_connected(net, 40)
            net = tflearn.fully_connected(net, 40)
            net = tflearn.fully_connected(net, len(self.output[0]),activation="softmax")
            net = tflearn.regression(net)

            self.model = tflearn.DNN(net)

            # self.model.fit(self.training, self.output, n_epoch=850, batch_size=8, show_metric=True)
            # self.model.save(f"./Category{self.category}/chatbot.tflearn")

        elif self.category == 5:
            net = tflearn.input_data(shape=[None, len(self.training[0])])
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, 50)
            net = tflearn.fully_connected(net, len(self.output[0]),activation="softmax")
            net = tflearn.regression(net)

            self.model = tflearn.DNN(net)

            # self.model.fit(self.training, self.output, n_epoch=500, batch_size=8, show_metric=True)
            # self.model.save(f"./Category{self.category}/chatbot.tflearn")



        # try:
        self.model.load(f"./Category{self.category}/chatbot.tflearn")
        #     self.loaded = pickle.load(open(f"./Category{self.category}/model.pkl", 'rb'))
        # except:
        # self.model.fit(self.training, self.output, n_epoch=500, batch_size=8, show_metric=True)
        # self.model.save(f"./Category{self.category}/chatbot.tflearn")
        # pickle.dump(self.model, open(f"./Category{self.category}/model.pkl", 'wb'))
        
        
    def _bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)


    def get_data(self, tag, lang):
        a = self.redisClient.hgetall(f'{self.category}:{tag}:{lang}')
        x = {y.decode('ascii'): a.get(y).decode('utf-8') for y in a.keys()}
        return x

    def prediction(self, s, lang):
        res = self.model.predict([self._bag_of_words(s,self.words)])[0]
        res_index = numpy.argmax(res)
        tag = self.labels[res_index]
        chain = False

        if res[res_index] > 0.7:
            responses = self.get_data(tag, lang)
            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    if tg["context_set"] is "1":
                        chain = True
                    else:
                        chain = False
            responses['chain'] = chain
            return responses
        else:
            a = self.redisClient.hgetall(f'custom:{lang}')
            x = {y.decode('ascii'): a.get(y).decode('utf-8') for y in a.keys()}
            x['chain'] = chain
            return x


app = Flask(__name__)
category = {'en': [
    {"category_id":1, "category_name":"Symptoms"},
    {"category_id":2, "category_name":"Modes of disease spread"},
    {"category_id":3, "category_name":"Precautions"},
    {"category_id":4, "category_name":"Cures"},
    {"category_id":5, "category_name":"About Corona"}
],'kn': [
    {"category_id":1, "category_name":"ಲಕ್ಷಣಗಳು"},
    {"category_id":2, "category_name":"ರೋಗ ಹರಡುವ ವಿಧಾನಗಳು"},
    {"category_id":3, "category_name":"ಮುನ್ನಚ್ಚರಿಕೆಗಳು"},
    {"category_id":4, "category_name":"ರೋಗಪರಿಹಾರ"},
    {"category_id":5, "category_name":"ಕರೋನಾ ಬಗ್ಗೆ"}
],'ta': [
    {"category_id":1, "category_name":"அம்சங்கள்"},
    {"category_id":2, "category_name":"நோய் பரவல் முறைகள்"},
    {"category_id":3, "category_name":"முன்னறிவிப்புகள்"},
    {"category_id":4, "category_name":"நோய் கண்டறிதல்"},
    {"category_id":5, "category_name":"கொரோனா பற்றி"}
],'te': [
    {"category_id":1, "category_name":"లక్షణాలు"},
    {"category_id":2, "category_name":"వ్యాధి వ్యాప్తి యొక్క రీతులు"},
    {"category_id":3, "category_name":"భవిష్యత్"},
    {"category_id":4, "category_name":"వ్యాధి ఉపశమనం"},
    {"category_id":5, "category_name":"కరోనా గురించి"}
],'ml': [
    {"category_id":1, "category_name":"സവിശേഷതകൾ"},
    {"category_id":2, "category_name":"രോഗം പടരുന്ന രീതി"},
    {"category_id":3, "category_name":"മുൻകരുതലുകൾ"},
    {"category_id":4, "category_name":"സുഖപ്പെടുത്തുന്നു"},
    {"category_id":5, "category_name":"കൊറോണയെക്കുറിച്ച്"}
],'hi' : [
    {"category_id":1, "category_name":"विशेषताएं"},
    {"category_id":2, "category_name":"रोग फैलने के तरीके"},
    {"category_id":3, "category_name":"एहतियात"},
    {"category_id":4, "category_name":"इलाज"},
    {"category_id":5, "category_name":"कोरोना के बारे में"}
]}
language = [
    {'lang_code': 'en', 'lang_name':'English'},
    {'lang_code': 'kn', 'lang_name':'Kannada'},
    {'lang_code': 'te', 'lang_name':'Telugu'},
    {'lang_code': 'ta', 'lang_name':'Tamil'},
    {'lang_code': 'ml', 'lang_name':'Malayalam'},
    {'lang_code': 'hi', 'lang_name':'Hindi'}
]

# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.args.get('token')
# 
#         if not token:
#             return jsonify({'msg': 'Token missing'}), 403
# 
#         try:
#             data = jwt.decode(token, app.config['SECRET_KEY'])
#         except:
#             return jsonify({'msg': 'Token invalid'}), 403
# 
#         return f(*args, **kwargs)
#     return decorated



# @app.route('/login', methods = ['GET'])
# def login():
#     auth = request.authorization

#     if auth and auth.password == '12345':
#         token = jwt.encode({'user' : auth.username}, app.config['SECRET_KEY'])

#         return jsonify({'token': token.decode('UTF-8'), "category":category})

#     return make_response('Could not Verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required"'})

@app.route('/category', methods = ['GET'])
def category_list():
    try:
        lang = request.args.get('lang')
    except:
        reply = {"Error": "Missing Language Arguement"}
        return jsonify(reply), 400
    try:
        return jsonify({'category': category[lang]}), 200
    except:
        reply = {"Error": "Internal Server Error, Please contact devs"}
        return jsonify(reply), 500


@app.route('/language', methods = ['GET'])
def language_list():
    try:
        return jsonify({'language': language}), 200
    except:
        reply = {"Error": "Internal Server Error, Please contact devs"}
        return jsonify(reply), 500



@app.route('/<int:ch>', methods=['POST'])
# @token_required
def send_string(ch):
    try:
        some_json = request.get_json()
        st = some_json["ques"]
        lang = some_json['lang']
        word_l = ["corona virus", "corona", 'Corona virus', 'Corona', 'coronavirus','coronavirus', 'COVID-19', 'COVID19', 'COVID 19', 'covid', 'karuna', 'covid19', 'korona']
        for i in word_l:
            st = st.replace(i ,"virus")
        st = st.replace("?", "")
        st = st.replace(".", "")
        st = st.replace('`', "")
        st = st.replace("'", "")

    except:
        reply = {"Error": "Json format error"}
        return jsonify(reply), 400   
    # try:
    if ch in list(range(1,6)):
        chatbot = Chatbot(ch)
        reply = chatbot.prediction(st.lower(), lang)
        return jsonify(reply), 200
    else:
        reply = {"Error": "Invalid category"}
        return jsonify(reply), 501
    # except:
    #     reply = {"Error": "Internal Server Error, Please contact devs"}
    #     return jsonify(reply), 500


        

if __name__ == "__main__":
    app.run(debug=True)