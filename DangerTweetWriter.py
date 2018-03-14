from nltk.twitter import Streamer, TweetWriter;
import os, datetime;

"""
This class is an extension of the TweetWriter from NLTK, which is covered under the Apache License version 2.0.
See: NLTK_LICENSE.txt.

This class ignores retweets, and seeks the phrases words listed in dangerPhrases. If the tweet does not contain one of those phrases, it is ignored.
"""
class DangerTweetWriter(TweetWriter):
    dangerPhrases = ["hate life", "kill myself", "suicide", "fuck life", "not worth living", "want to die", "hate myself", "hate everything", "depressed", "depression", "everything is gray", "no one cares", "will miss me", "want to disappear", "do not think I'll be at school next week","makes things easier for everyone"];

    def __init__(self, lim = 100):
        TweetWriter.__init__(self, limit=lim, subdir="Tweets");

    def timestamped_file(self):
        """
        :return: timestamped file name
        :rtype: str
        """
        subdir = "./Tweets/"##self.subdir
        fprefix = self.fprefix
        if subdir:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

        fname = os.path.join(subdir, fprefix)
        fmt = '%Y%m%d-%H%M%S'
        timestamp = datetime.datetime.now().strftime(fmt)
        if self.gzip_compress:
            suffix = '.gz'
        else:
            suffix = ''
        outfile = '{0}.{1}.txt{2}'.format(fname, timestamp, suffix)
        return outfile

    def handle(self, data):

        """
        Write Twitter data as ONLY DATA

        :return: return `False` if processing should cease, otherwise return `True`.
        :param data: tweet object returned by Twitter API
        """

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
            if self.gzip_compress:
                self.output.write("<START>" + (data['text'] + "<END>\n").encode('utf-8'))
            else:
                self.output.write("<START>" + data['text'] + "<END>\n")
        else:
            ##print("Retweet or non-useful tweet.");
            self.counter -= 1;

        self.check_date_limit(data)
        if self.do_stop:
            return

        self.startingup = False
