from nltk.tokenize import TweetTokenizer;
import nltk;
import operator;
import math;
import pickle;
import random;
from TweetTester import TweetClassifier;
from nltk.classify.scikitlearn import SklearnClassifier;
from sklearn.naive_bayes import MultinomialNB,BernoulliNB;
from sklearn.linear_model import LogisticRegression,SGDClassifier;
from sklearn.svm import SVC, LinearSVC, NuSVC;
from nltk.classify import ClassifierI;

TOP_WORDS_COUNT = 2000;
TRAINING_PERCENT = .8;
TWEETS_TO_LOAD = 4000;

tokenizer = TweetTokenizer(strip_handles = True);

## Build a list of every word in our tweets:

print("Classifying tweets and building word list.");
inputFile = open("./ClassifiedTweets.txt", "r");

allWords = [];
## 'P': [], -- REMOVED POSITIVE - testing
tweets = {'N': [], 'S': []};

count = 0;
classifier = '';
ctweet = [];
for line in inputFile:
    line = line.rstrip();
    if (classifier == ''):
        classifier = line[1:2];

        ## Change all Ps to Neutral.
        if (classifier == 'P'):
            classifier = 'N';

        line = line[10:];

    endLoc = line.find("<END>");
    if (endLoc > -1):
        line = line[:endLoc];

        tokenLine = tokenizer.tokenize(line);
        for token in tokenLine:
            token = token.lower();
            allWords.append(token);
            ctweet.append(token);

        ##print(classifier);
        ##print(ctweet);
        tweets[classifier].append(ctweet);

        classifier = '';
        ctweet = [];
        count = count + 1;
    else:
        tokenLine = tokenizer.tokenize(line);
        for token in tokenLine:
            token = token.lower();
            allWords.append(token);
            ctweet.append(token);

    if (count > TWEETS_TO_LOAD):
        break;

inputFile.close();

print("Word list building complete, selecting top", TOP_WORDS_COUNT, "words.");

allWords = nltk.FreqDist(allWords);

## So, I think the below is flawed. We presumably want the top 3000 most frequent, not just the first 3000 that show up. This seems to
## actually not do anything, when comparing the list pre-freqDist and after.
##wordFeatures = list(allWords.keys())[:TOP_WORDS_COUNT];
## So:
sortedFeatures = sorted(allWords.items(), key=operator.itemgetter(1), reverse = True);

sfl = len(sortedFeatures);

wordFeatures = [];

for i in range(TOP_WORDS_COUNT):
    if (i < sfl):
        wordFeatures.append(sortedFeatures[i][0]);

topWords = open("./Models/TopWords.txt", "w");
for word in wordFeatures:
    topWords.write(word + "\n");
topWords.close();

## Now we have the actual TOP_WORDS_COUNT words stored. We will save this later, when we start saving pickles.

##print(wordFeatures);

def findFeatures(tweet):
    words = set(tweet);
    features = {};
    for w in wordFeatures:
        features[w] = (w in words);

    return features;

##print((findFeatures(tweets['P'][0])));

print("Building training set.");

##tweets = {'P': [], 'N': [], 'S': []};
##featuresets = [(find_features(rev), category) for (rev, category) in documents]
##featureSets = [(findFeatures(tweet), category) for category in ['P', 'N', 'S'] for tweet in tweets[category]];
featureSets = [(findFeatures(tweet), category) for category in ['S', 'N'] for tweet in tweets[category]];

random.shuffle(featureSets);

featureSetCount = len(featureSets);
trainingSetCount = math.floor(featureSetCount * TRAINING_PERCENT);
print("Number of feature sets:", featureSetCount);
print("Number used for training:", trainingSetCount);

trainingSet = featureSets[:trainingSetCount];
testingSet = featureSets[trainingSetCount:];

#################################
## Build the Classifiers
################################

print("Training Naive Bayes");
naiveBayes = nltk.NaiveBayesClassifier.train(trainingSet);
print("Training Multinomial NB");
MNB_classifier = SklearnClassifier(MultinomialNB());
MNB_classifier.train(trainingSet);
print("Training Bernoulli NB");
BernoulliNB_classifier = SklearnClassifier(BernoulliNB());
BernoulliNB_classifier.train(trainingSet);
print("Training Logistic Regression");
LogisticRegression_classifier = SklearnClassifier(LogisticRegression());
LogisticRegression_classifier.train(trainingSet);
print("Training SGDC");
SGDClassifier_classifier = SklearnClassifier(SGDClassifier());
SGDClassifier_classifier.train(trainingSet);
print("Training SVC");
SVC_classifier = SklearnClassifier(SVC());
SVC_classifier.train(trainingSet);
print("Training Linear SVC");
LinearSVC_classifier = SklearnClassifier(LinearSVC());
LinearSVC_classifier.train(trainingSet);

naiveBayesF = open("./Models/naivebayes.pickle","wb");
MNBF = open("./Models/MNB.pickle","wb");
BNBF = open("./Models/BNB.pickle","wb");
LRF = open("./Models/LR.pickle","wb");
SGDCF = open("./Models/SGDC.pickle","wb");
SVCF = open("./Models/SVC.pickle","wb");
linSVCF = open("./Models/LinSVC.pickle","wb");

pickle.dump(naiveBayes, naiveBayesF);
pickle.dump(MNB_classifier, MNBF);
pickle.dump(BernoulliNB_classifier, BNBF);
pickle.dump(LogisticRegression_classifier, LRF);
pickle.dump(SGDClassifier_classifier, SGDCF);
pickle.dump(SVC_classifier, SVCF);
pickle.dump(LinearSVC_classifier, linSVCF);

naiveBayesF.close();
MNBF.close();
BNBF.close();
LRF.close();
SGDCF.close();
SVCF.close();
linSVCF.close();

tweetClassifier = TweetClassifier();

naiveBayes.show_most_informative_features(15);

accuracyWriter = open("./Models/AccuracyPerAlgorithm.txt", "w");

def printAndWrite(*args):
    ##print(args);
    for a in args:
        accuracyWriter.write(str(a));
        accuracyWriter.write(" ");
        print(a, end = " ");
    accuracyWriter.write("\n");
    print();

print("Testing accuracies, some of these may be slow: ");
printAndWrite("Naive Bayes accuracy percent:", (nltk.classify.accuracy(naiveBayes, testingSet))*100);
printAndWrite("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testingSet))*100);
printAndWrite("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testingSet))*100);
printAndWrite("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testingSet))*100);
printAndWrite("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testingSet))*100);
printAndWrite("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testingSet))*100);
printAndWrite("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testingSet))*100);
print("Testing voted classifier, this is somewhat slow: ");
printAndWrite("Voted Classifier accuracy percent:", tweetClassifier.testAccuracy(testingSet));


accuracyWriter.close();
