import json
import os
import sys
import unittest
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter import app

PAYLOAD_TWEETS_EN = """
    [
        {"text": "@MyTeleC today is the first day for my billing period. I had mobile data turned off since last night. How did I loose 500kbyes? https://t.co/VUZV4moYhr"},
        {"text": "@MyTeleC I tried Vodafones broadband... Constant drop outs made it unusable, speed was ok when it worked.I went crawling back to sky within 2 months...The best thing about Vodafone broadband is its easy to cancel!"},
        {"text": "@MyTeleC why isit impossible to speak to one of your advisors and when I do manage to get through to someone im cut of. Im trying to pay my bill for god sake. Do you want it or not?!"},
        {"text": "@MyTeleC hey just wondering do you offer 4G calling in these two postcodes ng175be and this one Ng174ad? Can't find out on the site"},
        {"text": "@MyTeleC I understand this and was told last night, I can terminate from the end of June. You really don't understand customer loyalty/retention. Again, shocking"},
        {"text": "@MyTeleC @JohnBoyega @touchithighnote the vid"}
    ]
"""

PAYLOAD_TWEETS_IT = """
    [
        {"text": "@Op1 buongiorno ho scritto ieri un dm nessuna risposta!"},
        {"text": "@Op1 voglio dire grazie pubblicamente al vostro operatore xxxxxxx (lavorava ieri pomer.) che in due minuti ha risolto tutto. Tks"},
        {"text": "@Op1 buongiorno sto partendo per le Grecia, posso utilizzare il mio piano tariffario senza spendere nulla? nGrazie"},
        {"text": "@Op1 √à prevista qualche offerta con almeno 10GB con uno costo di attivazione inferiore?"},
        {"text": "Cara @Op1 impossibile da tempo parlare con vs operatore e sto pagando giochi nn richiesti ¬†nPasso alla concorrenza o mi contattate subito?"},
        {"text": "@Op1 sono senza linea (cellulare) da ieri.Posso usare solo il wifi di casa ma non posso n√© chiamare n√© ricevere. Numero mandato in DM"}
    ]
"""


class TestBasic(unittest.TestCase):
    def setUp(self):
        # Ignore third-party warnings from setup phase..
        warnings.simplefilter('ignore')
        # Initialize Flask server hosting api
        self.app = app.test_client()

    def test_post_classification_result(self):
        #################################
        # Test English tweet classifier #
        #################################
        resp = self.app.post(
            '/hitec/classify/domain/tweets/lang/en',
            data=PAYLOAD_TWEETS_EN
        )
        
        status_code = resp._status_code
        body = resp.get_data()
        self.assertEqual(status_code, 200, msg='API should be responsive with 200 status code')
        
        for tweet in json.loads(body):
            self.assertEqual("tweet_class" in tweet, True, msg='classified tweets should contain the defined key')


        #################################
        # Test Italian tweet classifier #
        #################################
        resp = self.app.post(
            '/hitec/classify/domain/tweets/lang/it',
            data=PAYLOAD_TWEETS_IT
        )
        
        status_code = resp._status_code
        body = resp.get_data()
        self.assertEqual(status_code, 200, msg='API should be responsive with 200 status code')

        for tweet in json.loads(body):
            self.assertEqual("tweet_class" in tweet, True, msg='classified tweets should contain the defined key')

if __name__ == '__main__':
    # Run the TestBasic unit tests
    unittest.main(verbosity=2)
