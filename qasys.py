import sys
import codecs
from nltk.stem.snowball import SnowballStemmer
from nltk.tag.stanford import StanfordNERTagger
import nltk
import operator
import math
import nltk.data

class Question:
    def __init__(self, number, qType, tokens, weights):
        self.number = number
        self.qType = qType
        self.tokens = tokens
        self.weights = weights

TRAIN_Q_PATH = "hw5data/qadata/train/questions.txt"
TEST_Q_PATH = "hw5data/qadata/test/questions.txt"
STOP_WORDS_PATH = "hw5data/stopwords.txt"
TRAIN_TOPDOCS_PATH = "hw5data/topdocs/train/top_docs."
TEST_TOPDOCS_PATH = "hw5data/topdocs/test/top_docs."

SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
NER_TAGGER = StanfordNERTagger('/home/cs-students/18bfd2/bin/stanford-ner-2014-08-27/classifiers/english.muc.7class.distsim.crf.ser.gz', '/home/cs-students/18bfd2/bin/stanford-ner-2014-08-27/stanford-ner.jar')

predictionFile = open('prediction.txt', 'w')

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
        questions.append(Question(int(questionLines[i].split()[1]), processedQ[0], processedQ[1], processedQ[2]))
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

    weights = [1.0 for x in range(0, len(tokens))]
    return (qType, list(tokens), weights)
    
#Retrieve a set of stopwords
def getStopWords():
    with codecs.open(STOP_WORDS_PATH, 'r', 'cp437') as stopWordData:
        return set(stopWordData.read().split())

#Retrive a list of the tokens in the topdocs for associated question
def getTrainTopDocs(qNum):
    with codecs.open(TRAIN_TOPDOCS_PATH + str(qNum), 'r', 'cp437') as topdocsData:
        sentences =  SENT_DETECTOR.tokenize(topdocsData.read().strip())
    return [x.split() for x in sentences]

#Retrive a list of the tokens in the topdocs for associated question
def getTestTopDocs(qNum):
    with codecs.open(TEST_TOPDOCS_PATH + str(qNum), 'r', 'cp437') as topdocsData:
        sentences =  SENT_DETECTOR.tokenize(topdocsData.read().strip())
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

    #In the case of 0 magnitude, we want the similarity to be zero
    if denominator == 0:
        return 0 
    else:
        return (numerator / denominator)

#Find average distance to question keywords
def keywordDistance(docSentence, question, ansIndex):
    keywordIndices = []
    for i in range(0, len(docSentence)):
        if docSentence[i].lower() in question.tokens:
            keywordIndices.append(i)

    invResult = 0.0
    for i in range(0, len(keywordIndices)):
        invResult += math.sqrt(math.abs(ansIndex - i))
    
    return 1.0/invResult

#For a given document sentence and set of query keywords, return a vector
# that represents the frequency with which those keywords occury in the
# sentence
def getQueryVector(docSentence, query):
    keywordVector = [0 for x in range(0, len(query))]
    for token in docSentence:
        if token.lower() in query:
            keywordVector[query.index(token.lower())] += 1
    return keywordVector

#Sort the top docs for a given question by their cosine similarity to 
# the keywords of the question.
def sortTopDocs(sentences, question):
    sentences.sort(key=lambda x: cosSim(question.weights, getQueryVector(x, question.tokens)), reverse=True)

#Method to edit most
#Takes the top 5 documents and ranks which entities might be answers
def extractAnswers(sortedDocs, question):
    possibleAnswers = []
    weights = []
    nounTags = ['NN', 'NNP', 'NNS', 'NNPS']

    #Take top 5 topdocs

    if question.qType is 'where' or question.qType is 'who':
        for i in range(0, 5):
            namedEntities = NER_TAGGER.tag(sortedDocs[i])
            print str(namedEntities)
            
            #Consider each entity tagged
            for j in range(0, len(namedEntities)):
                entity = namedEntities[j]

                #If the entity is not a named entity or is part of the question, don't consider it
                if not (entity[1] == 'O' or entity[0] in question.tokens):

                #If the entity is not yet a possible answer add it, else increment it's weight
                    if not (entity[0] in possibleAnswers):
                        possibleAnswers.append(entity[0])
                        weights.append(1)
                    else:
                        weights[possibleAnswers.index(entity[0])] += 1
    elif question.qType is 'when' or 'how':
        for i in range(0, 5):
            partsOfSpeech = nltk.pos_tag(sortedDocs[i])
            print str(partsOfSpeech)

            #Consider each entity tagged
            for j in range(0, len(partsOfSpeech)):
                entity = namedEntities[j]
                
                #If the entity is not a named entity or is part of the question, don't consider it
                if  entity[1] is 'CD' and not entity[0] in question.tokens:

                #If the entity is not yet a possible answer add it, else increment it's weight
                    if not (entity[0] in possibleAnswers):
                        possibleAnswers.append(entity[0])
                        weights.append(1)
                    else:
                        weights[possibleAnswers.index(entity[0])] += 1
    elif question.qType is 'what' or 'which' or 'why' or 'unknown':
    return sorted(possibleAnswers, key=lambda x: weights[possibleAnswers.index(x)])

def writeAnswersToFile(possibleAnswers, qid):
    predictionFile.write('qid ' + str(qid) + '\n')
    for i in range(0, min(len(possibleAnswers), 10)):
        predictionFile.write(possibleAnswers[i] + '\n')
            
if __name__ == "__main__":
    #Read in data
    trainingQs = getQuestions(TRAIN_Q_PATH)
    testQs = getQuestions(TEST_Q_PATH)

    #Attempt to answer training questions
    for q in trainingQs:
        print str(q.number) + "================================="
        print str(q.qType)
        for token in q.tokens:
            print token + " "

        topDocs = getTrainTopDocs(q.number)
        sortTopDocs(topDocs, q)
        for i in range(0, 5):
            print "\nTopDoc number " + str(i + 1) + ": " + str(topDocs[i])

        possibleAnswers = extractAnswers(topDocs, q)
        writeAnswersToFile(possibleAnswers, q.number)
    predictionFile.close()
    

