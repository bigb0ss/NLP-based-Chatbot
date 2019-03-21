import nltk
import warnings
warnings.filterwarnings("ignore")
from flask import Flask , render_template , request
import string
import random


f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


sent_tokens[:2]


word_tokens[:5]


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    global LemNormalize
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
	user_response = request.form['input']
	
	user_response=user_response.lower()
	output=response(user_response)
	#return render_template('index.html',output=output)
	if(user_response!='bye'):
		if(user_response=='thanks' or user_response=='thank you' ):
			flag=False
			return render_template('index.html',output=" You are welcome..")
		else:
			if(greeting(user_response)!=None):
				return render_template('index.html',output=" "+greeting(user_response))
			else:
				return render_template('index.html',output=output)
				sent_tokens.remove(user_response)
	else:
		flag=False
		return render_template('index.html',output=" Bye! take care..")


if __name__=='__main__':
	f=open('chatbot.txt','r',errors = 'ignore')
	raw=f.read()
	raw=raw.lower()# converts to lowercase
	#nltk.download('punkt') # first-time use only
	#nltk.download('wordnet') # first-time use only
	sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
	word_tokens = nltk.word_tokenize(raw)# converts to list of words
	sent_tokens[:2]
	word_tokens[:5]
	lemmer = nltk.stem.WordNetLemmatizer()
	remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
	GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
	GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
	app.run(debug=True)