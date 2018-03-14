from nltk.classify import ClassifierI
from nltk.tokenize import TweetTokenizer;
import pickle;
import nltk;
from statistics import mode

## The vote classifier developed in TCSS 456:
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

## A class that uses all the pickles, as well as the list of top features, to try to classify tweets.
class TweetClassifier():
    def __init__(self, *args):
        ## Load all saved classifiers.
        naiveBayesF = open("./Models/naivebayes.pickle","rb");
        MNBF = open("./Models/MNB.pickle","rb");
        BNBF = open("./Models/BNB.pickle","rb");
        LRF = open("./Models/LR.pickle","rb");
        SGDCF = open("./Models/SGDC.pickle","rb");
        SVCF = open("./Models/SVC.pickle","rb");
        linSVCF = open("./Models/LinSVC.pickle","rb");

        naiveBayes = pickle.load(naiveBayesF);
        MNB = pickle.load(MNBF);
        BNB = pickle.load(BNBF);
        LR = pickle.load(LRF);
        SGDC = pickle.load(SGDCF);
        SVC = pickle.load(SVCF);
        linSVC = pickle.load(linSVCF);

        naiveBayesF.close();
        MNBF.close();
        BNBF.close();
        LRF.close();
        SGDCF.close();
        SVCF.close();
        linSVCF.close();

        ## Build a vote classifier.
        self.voteClassifier = VoteClassifier(naiveBayes, MNB, BNB, LR, SGDC, SVC, linSVC);

        self.NB = naiveBayes;
        self.MNB = MNB;
        self.BNB = BNB;
        self.LR = LR;
        self.SGDC = SGDC;
        self.SVC = SVC;
        self.linSVC = linSVC;


        self.wordFeatures = [];

        topWordsF = open("./Models/TopWords.txt");
        for line in topWordsF:
            self.wordFeatures.append(line.rstrip());
        topWordsF.close();

        self.tokenizer = TweetTokenizer(strip_handles = True);

    def findFeatures(self, tweetTokenized):
        words = set(tweetTokenized);
        features = {};
        for w in self.wordFeatures:
            features[w] = (w in words);

        return features;

    def testTweet(self, tweet):
        tweetTokens = self.tokenizer.tokenize(tweet);
        features = self.findFeatures(tweetTokens);

        classification = self.voteClassifier.classify(features);
        '''print("TESTING!");
        print(self.NB.classify(features));
        print(self.MNB.classify(features));
        print(self.BNB.classify(features));
        print(self.LR.classify(features));
        print(self.SGDC.classify(features));
        print(self.SVC.classify(features));
        print(self.linSVC.classify(features));'''

        confidence = self.voteClassifier.confidence(features)*100;

        return classification, confidence;

    def testAccuracy(self, trainingSet):
        return (nltk.classify.accuracy(self.voteClassifier, trainingSet))*100;
