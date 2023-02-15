from utils.general import load_lstm, predict_text,top_three_words
from utils.twitter_scraper import scraper
from flask import Flask, render_template, request, url_for
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import spacy

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
model,vocab_dict = load_lstm()
stopwords = list(STOP_WORDS)

@app.route('/', methods=['GET', 'POST'])
def index():
    lenpre = 0
    
    if request.method == 'POST':
        keyword = request.form['keyword']
        tweets = scraper(keyword=keyword)
        current_tweets = tweets[lenpre:]

        bad = []
        good = []
        positive_data = 0
        negative_data = 0
        keylist = []

        keylist = keyword.split(" ") + [keyword.replace(" ","")]
        

        for tw in current_tweets:
            
            if predict_text(tw.tweet,model,vocab_dict):
                positive_data += 1
                for token in nlp(tw.tweet):
                    if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.pos_ != 'SYM' and token.text.lower() not in keylist:
                        good.append(token.lemma_.strip())
            else:
                negative_data += 1
                for token in nlp(tw.tweet):
                    if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.pos_ != 'SYM' and token.text.lower() not in keylist :
                        bad.append(token.lemma_.strip())
 

        topbad = top_three_words(bad)
        topgood = top_three_words(good)

        lenpre = len(tweets)

        total_data = positive_data + negative_data
        # just to see len
        print(total_data,len(tweets),"---"*50)
        # Calculate percentages
        positive_percent = positive_data / total_data * 100
        negative_percent = negative_data / total_data * 100

        plt.clf()
        # create histogram based on keyword input
        # values = [len(data[keyword]) if keyword in data else 0 for k in data.keys()]
        plt.bar(['Positive', 'Negative'], [positive_percent, negative_percent])
        plt.ylabel('Percentage')
        # convert plot to base64 string for display in template
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        return render_template('index.html', plot_url=plot_url, topgood=topgood, topbad=topbad)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
