import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def printing_sentences_in_lowercase():
    # Converting the words of each sentence to a lowercase words and printing it
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    for sentence in sentences:
        sentence = sentence.lower()
        print(sentence)


def printing_sentences_in_uppercase():
    # Converting the words of each sentence to uppercase words and printing it
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    for sentence in sentences:
        sentence = sentence.upper()
        print(sentence)


def removing_punctuation_in_each_sentence():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for sentence in sentences:
        print("\nBefore removing punctuation: "+sentence)
        for word in sentence:
            if word in punctuations:
                sentence = sentence.replace(word, "")
        print("After removing punctuation: " + sentence)


# Tokenizing Sentences (used in different functions)
def tokenizing_sentences():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence_words = word_tokenize(sentence)
        tokenized_sentences.append(sentence_words)
    return tokenized_sentences


def print_tokenized_sentences():
    tokenized_sentences = tokenizing_sentences()
    for tokenized_sentence in tokenized_sentences:
        print(tokenized_sentence)


def removing_stopwords():
    # gets the stop words found in english in a set
    stops = set(stopwords.words('english'))
    tokenized_sentences = tokenizing_sentences()
    # Removes the stop words found in a sentence and printing the words in a list without the stop words
    for sentence in tokenized_sentences:
        print("\nBefore removing stop words: ", sentence)
        for word in sentence:
            if word in stops:
                sentence.remove(word)
        print("After removing stop words: ", sentence)


def stemming_words_in_each_sentence():
    stemmer = PorterStemmer()
    tokenized_sentences = tokenizing_sentences()

    for sentenceWords in tokenized_sentences:
        stemmed_sentence = []
        print("\nwords before stemming: ", sentenceWords)
        for word in sentenceWords:
            stemmed_sentence.append(stemmer.stem(word))
        print("words after stemming: ", stemmed_sentence)


def lemmatizing_words_in_each_sentence():
    lemmatizer = WordNetLemmatizer()
    tokenized_sentences = tokenizing_sentences()

    for sentence in tokenized_sentences:
        lemmatized_sentence = []
        print("\nwords before lemmatization: ", sentence)
        for word in sentence:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        print("words after lemmatization: ", lemmatized_sentence)


def sentiment_analysis():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    for sentence in sentences:
        print('\n'+sentence)
        print(sia.polarity_scores(sentence))


def classification():
    # Getting the Sentences from the excel file
    data = pd.read_csv('TextDataset.csv')
    # Adding all the sentences to a list
    sentences = list(data['text'])
    train = [("what an amazing weather", 'pos'),
             ("this is an amazing idea!", 'pos'),
             ("I feel very good about these ideas.", 'pos'),
             ("what an awesome view", 'pos'),
             ("I really like the zoom app cause it allows us to be there for each other in a time like now when many things Are uncertain.", 'pos'),
             ("this is my best performance.", 'pos'),
             ('Great sacrament meeting. I am very grateful for this opportunity to listen from home occasionally.', 'pos'),
             ('I like it because you can have a host and share your screen and you can do cool thing.', 'pos'),
             ('Well, we walk and fly on the shoulders of Giants!', 'pos'),
             ('I do not like this place', 'neg'),
             ('I am tired of this stuff.', 'neg'),
             ('The speaker was hard to understand at times because of glitches.', 'neg'),
             ('I can\'t deal with all this tension', 'neg'),
             ('he is my sworn enemy!', 'neg'),
             ('my friends are horrible', 'neg'),
             ]
    test = [
        ("i cannot download the app.customer support can’t even help.", "neg"),
        ("could you please help to fix it.thank you for your help.", "pos"),
        ("trying to join zoom is more difficult than it should be.", "neg"),
        ("this is a great resource conference", "pos"),
        ("very clear reception with great sound!", "pos"),
    ]
  #  c1 = NaiveBayesClassifier(train)
    for sentence in sentences:
        print("\n"+sentence)
  #      print("Classified: "+c1.classify(sentence))
   # print("\nClassification Accuracy: ", c1.accuracy(test))


def main_function():
    available_functions = [
        '1- Printing words in each sentence in lowercase.',
        '2- Change words in each sentence in uppercase.',
        '3- Remove punctuations in each sentence.',
        '4- Tokenizing each Sentence.',
        '5- Removing stop words.',
        '6- Stem words in each sentence.',
        '7- Lemmatize words in each sentence.',
        '8- Sentiment analysis',
        '9- Classification',
        '10- Exit Program.',
    ]
    for function in available_functions:
        print(function)

    function_number = int(input())
    if 1 <= function_number <= 10:
        return function_number
    else:
        print("*****Please enter a valid number*****")
        return main_function()


functionNumber = main_function()

while not functionNumber == 10:
    if functionNumber == 1:
        printing_sentences_in_lowercase()
    elif functionNumber == 2:
        printing_sentences_in_uppercase()
    elif functionNumber == 3:
        removing_punctuation_in_each_sentence()
    elif functionNumber == 4:
        print_tokenized_sentences()
    elif functionNumber == 5:
        removing_stopwords()
    elif functionNumber == 6:
        stemming_words_in_each_sentence()
    elif functionNumber == 7:
        lemmatizing_words_in_each_sentence()
    elif functionNumber == 8:
        sentiment_analysis()
    elif functionNumber == 9:
        classification()
    print('\n\n')
    functionNumber = main_function()

