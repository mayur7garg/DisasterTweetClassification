import flask
import string
import nltk
import re
import pandas as pd
import dill

wn = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
hashtag = r"#[a-zA-Z]\w+"

def tokenize_text(text):
    text = text.lower()
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('[^a-z]+', text)
    return tokens

with open('Models/kw_oe.pkl', 'rb') as file:
    kw_oe = dill.load(file)

with open('Models/tfidf.pkl', 'rb') as file:
    tfidf_vect = dill.load(file)

with open('Models/rf.pkl', 'rb') as file:
    rf = dill.load(file)

app = flask.Flask(__name__, template_folder='Templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST': 
        tweet = flask.request.form['tweet'].strip()[:280]

        keyword = flask.request.form['keyword'].strip().split()
        if len(keyword) < 1 or keyword[0] not in kw_oe.categories_[0]:
            keyword = 'NaN'
        else:
            keyword = keyword[0]

        kw_id = kw_oe.transform([[keyword]])[0][0]
        hashtag_count = len(re.findall(hashtag, tweet))
        text_len = len(tweet) - tweet.count(" ")
        text = tokenize_text(tweet)
        word_count = len(text)

        tf_df = pd.DataFrame(tfidf_vect.transform([text]).toarray(), columns = ['tf_' + colname for colname in tfidf_vect.get_feature_names()])
        input_df = pd.DataFrame([[kw_id, hashtag_count, text_len, word_count]], columns = ['keyword', 'hashtag_count', 'text_len', 'word_count'])
        input_df = pd.concat([input_df, tf_df], axis = 1)

        prediction =  rf.predict(input_df)[0]
        result = 'YES' if prediction == 1 else 'NO'

        return flask.render_template('main.html', tweet = tweet, result = result)
    
if __name__ == '__main__':
    app.run()