import sys
import codecs
from nltk.stem.snowball import SnowballStemmer
import nltk
import operator
import math
import nltk.data

class Question:
    def __init__(self, number, qType, tokens):
        self.number = number
        self.qType = qType
        self.tokens = tokens

#Usage

TRAIN_Q_PATH = "hw5data/qadata/train/questions.txt"
TEST_Q_PATH = "hw5data/qadata/test/questions.txt"
STOP_WORDS_PATH = "hw5data/stopwords.txt"
TRAIN_TOPDOCS_PATH = "hw5data/topdocs/train/top_docs."
TEST_TOPDOCS_PATH = "hw5data/topdocs/test/top_docs."

SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

#Read in and process the questions located at path
def getQuestions(path):

    #Grab data
    with codecs.open(TRAIN_Q_PATH, 'r', 'cp437') as  questionData:
        questionLines = questionData.read().split('\n')

    questions = []
    stopWords = getStopWords()

    #Pack questions into Question object
    for i in range(0, len(questionLines) - 2, 3):
        processedQ = processQuestion(questionLines[i + 1].split(), stopWords)
        questions.append(Question(int(questionLines[i].split()[1]), processedQ[0], processedQ[1]))
    return questions

#Process a question to determine its type and prepare it's query
def processQuestion(tokens, stopWords):
    #Strip casing
    tokens = set([x.lower() for x in tokens])

    #Determine question type
    qTypes = ["where", "who", "when", "what", "why", "which", "how"]
    qType = "unknown"
    for x in qTypes:
        if x in tokens:
            qType = x
            break

    #Strip casing and stopwords
    tokens = tokens - stopWords
    punctuation = [".", ",", "\"", "\'", "?", "!"]
    for p in punctuation:
        tokens = set([x.replace(p, "") for x in tokens])

    return (qType, tokens)
    
#Retrieve a set of stopwords
def getStopWords():
    with codecs.open(STOP_WORDS_PATH, 'r', 'cp437') as stopWordData:
        return set(stopWordData.read().split())

#Retrive a list of the tokens in the topdocs for associated question
def getTrainTopDocs(qNum):
    with codecs.open(TRAIN_TOPDOCS_PATH + str(qNum), 'r', 'cp437') as topdocsData:
        sentences = SENT_DETECTOR.tokenize(topdocsData.read().strip()) #CHANGE TO SENTENCE SPLIT
    return [x.split() for x in sentences]

#Retrive a list of the tokens in the topdocs for associated question
def getTestTopDocs(qNum):
    with codecs.open(TEST_TOPDOCS_PATH + str(qNum), 'r', 'cp437') as topdocsData:
        sentences =  SENT_DETECTOR.tokenize(topdocsData.read().strip()) #CHANGE TO SENTENCE SPLIT
    return [x.split() for x in sentences]

#Comptues magnitude of a vector. Helper for Cosine Similarity
def mag(v):
    magnitude = 0.0
    for x in v:
        magnitude += float(x * x)
    math.sqrt(magnitude)
    return magnitude

#Determine cosine similarity score between two vectors
def cosSim(vec1, vec2):
    numerator = float(sum([vec1[i] * vec2[i] for i in range(0, len(vec1))]))
    denominator = mag(vec1) * mag(vec2)
    if denominator == 0:
        return 0
    else:
        return (numerator / denominator)

#def getQueryVector(docSentence, query):
    

if __name__ == "__main__":
    #Read in data
    trainingQs = getQuestions(TRAIN_Q_PATH)
    testQs = getQuestions(TEST_Q_PATH)

    #Verify questions are being read correctly
    for q in trainingQs:
        print str(q.number) + "================================="
        print str(q.qType)
        for token in q.tokens:
            print token + " "

    print(getTrainTopDocs(1)[0])

