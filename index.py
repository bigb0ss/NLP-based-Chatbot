import nltk
import warnings
warnings.filterwarnings("ignore")
from flask import Flask , render_template , request
import string
import random

if True:
        f=open('chatbot.txt','r',errors = 'ignore')
        raw=f.read()
        raw=raw.lower()# converts to lowercase
        # first-time use only
        sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
        word_tokens = nltk.word_tokenize(raw)# converts to list of words
        lemmer = nltk.stem.WordNetLemmatizer()
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
        GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def LemTokens(tokens):
        global lemmer
        return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
        global remove_punct_dict
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Checking for greetings
def greeting(sentence):
        """If user's input is a greeting, return a greeting response"""
        for word in sentence.split():
                if word.lower() in GREETING_INPUTS:
                        return random.choice(GREETING_RESPONSES)
        return None


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
        global sent_tokens
        robo_response=''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
                robo_response=robo_response+"I am sorry! I don't understand you"
                return robo_response
        else:
                robo_response = robo_response+sent_tokens[idx]
                return robo_response


app=Flask(__name__)



@app.route('/')
def index():
        return render_template('index.html',output="My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

@app.route('/reply',methods=['GET','POST'])
def respond():
        global sent_tokens
        user_response = request.form['input']
	
        user_response=user_response.lower()
        output=response(user_response)
        #return render_template('index.html',output=output)
        if(user_response!='bye'):
                if(user_response=='thanks' or user_response=='thank you' ):
                        return render_template('index.html',output=" You are welcome..")
                else:
                        if(greeting(user_response)!=None):
                                return render_template('index.html',output=" "+greeting(user_response))
                        else:
                                sent_tokens.remove(user_response)
                                return render_template('index.html',output=output)
        else:
                return render_template('index.html',output=" Bye! take care..")


if __name__=='__main__':
        app.run()
