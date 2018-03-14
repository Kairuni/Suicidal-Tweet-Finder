print("Importing");
from DangerTweetWriter import DangerTweetWriter;
from nltk.twitter import Streamer, TweetWriter, credsfromfile;
from nltk.corpus import stopwords;
from TweetTester import TweetClassifier;
import os, datetime;

"""
This class is an extension of the TweetWriter from NLTK, which is covered under the Apache License version 2.0.
See: NLTK_LICENSE.txt.

This class ignores retweets, and seeks the phrases words listed in dangerPhrases. If the tweet does not contain one of those phrases,
"""
class LiveTweetClassifierAndWriter(DangerTweetWriter):
    def __init__(self, lim = 100):
        DangerTweetWriter.__init__(self, lim);
        self.tweetClassifier = TweetClassifier();

    def handle(self, data):
        if self.startingup:
            if self.gzip_compress:
                self.output = gzip.open(self.fname, 'w')
            else:
                self.output = open(self.fname, 'w')
            print('Writing to {0}'.format(self.fname))

        found = False;
        for phrase in self.dangerPhrases:
            if phrase in data['text']:
                found = True;

        if (data['text'][:2] != "RT" and found):
            print("Useful tweet:", data['text']);
            print("Testing classification: ");
            classification, confidence = self.tweetClassifier.testTweet(data['text']);

            print("Classified as",classification,"at",confidence,"% confidence.");

            if self.gzip_compress:
                self.output.write(("<"+classification+">"+ data['text'] + "\n").encode('utf-8'));
            else:
                self.output.write("<"+classification+">" + data['text'] + "\n");
        else:
            ##print("Retweet or non-useful tweet.");
            self.counter -= 1;

        self.check_date_limit(data)
        if self.do_stop:
            return

        self.startingup = False

print("Building auth");

oauth = credsfromfile();

twitterPass = "";
alreadyAdded = [];
stopWords = set(stopwords.words('english'));

for phrase in DangerTweetWriter.dangerPhrases:
    split = phrase.split(" ");
    for word in split:
        if (word not in alreadyAdded and word not in stopWords):
            if (len(alreadyAdded) > 0):
                twitterPass = twitterPass + ", ";
            twitterPass = twitterPass + word;
            alreadyAdded.append(word);

print("Words to pass to Twitter's filter: " + twitterPass);

client = Streamer(**oauth);
client.register(LiveTweetClassifierAndWriter(10000));
print("Attempting to stream");
client.filter(twitterPass);
