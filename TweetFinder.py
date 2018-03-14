print("Importing");
from DangerTweetWriter import DangerTweetWriter;
from nltk.twitter import Streamer, credsfromfile;

from nltk.corpus import stopwords;

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
client.register(DangerTweetWriter(10));
print("Attempting to stream");
client.filter(twitterPass);
