from sentence_transformers import SentenceTransformer
import spacy

MIN_SIMILARITY = 0.45

actions_dict = {
    "Hello":		["Hello!", "Hi!", "Good morning!", "Good afternoon!", "Good evening!", "Goodbye!", "See you!", "See you later!", "Take care!", "Welcome!", "Hi there!", "Bye!", "Bye-bye!", "Catch you later!", "Farewell!", "Have a good day!", "Have a good night!", "What's up?", "Howdy!", "Cheers!", "Nice to see you!", "Until next time!", "Peace out!", "So long!", "Safe travels!"],
    "Yes":			["Ok", "Yes", "Yeah", "Yep", "Yup", "Sure", "Absolutely", "Certainly", "Definitely", "Of course", "Indeed", "Affirmative", "Roger that", "For sure", "Totally", "You bet", "Right", "Alright"],
    "No":			["No", "Nope", "Nah", "Not at all", "Absolutely not", "Certainly not", "Definitely not", "No way", "Not really", "I don’t think so", "Negative", "No chance", "By no means", "Never", "No sir", "No ma'am", "Not on your life", "Not in a million years", "Out of the question", "Not happening", "Not possible", "No can do", "Nay", "I refuse"],
    "I dont know":	["Maybe", "Perhaps", "Not sure", "I don't know", "Unsure", "Possibly", "Could be", "Not certain", "Not clear", "It's unclear", "Can't say", "I guess", "I suppose", "Not positive", "Up in the air", "Doubtful", "Questionable", "Ambiguous", "Indecisive", "On the fence"],
    "Think":        ["Think", "Reflect", "Ponder", "Consider", "Contemplate", "Muse", "Deliberate", "Meditate", "Wonder", "Speculate", "Imagine", "Mull over", "Ruminate", "Analyze", "Envision", "Dream"],
    "Happy": 		["Applause", "Thumbs up", "Yay!", "Woohoo!", "Hooray!", "Bravo!", "Great job!", "Well done!", "Fantastic!", "Awesome!", "Excellent!", "Terrific!", "Amazing!", "Superb!", "Wonderful!", "Outstanding!", "Marvelous!", "Splendid!", "Kudos!", "High five!"],
    "Sad":          ["I am Sorry", "I am Sad", "I am unhappy", "I am heartbroken", "I am miserable", "I am sorrowful", "I am tearful", "I beg your pardon", "I am depressed"]
}

class Movement_recognition:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        #self.model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.actions_dict = {}
        for action, references in actions_dict.items():
            references_emb = []
            for reference in references:
                reference_emb = self.model.encode(reference)
                references_emb.append(reference_emb)
            self.actions_dict.update({action: references_emb})
        
        self.nlp = spacy.load('en_core_web_sm')
    

    def detect_actions(self, text):

        text = text.replace('\n', ' ').replace(',', '.')

        doc = self.nlp(text)

        sentences = []
        best_actions = []
        best_references = []
        best_similarities = []

        for sentence in doc.sents:
            sentence = sentence.text

            sentence_emb = self.model.encode(sentence)

            sentences.append(sentence)
            best_actions.append('Default')
            best_references.append(None)
            best_similarities.append(0)

            for action, references_emb in self.actions_dict.items():
                for index, reference_emb in enumerate(references_emb):
                    
                    similarity = self.model.similarity(sentence_emb, reference_emb).item()
                    
                    if similarity > best_similarities[-1]:
                        best_similarities[-1] = similarity
                        if similarity > MIN_SIMILARITY:
                            best_actions[-1] = action
                            best_references[-1] = actions_dict.get(action)[index]

        return sentences, best_actions, best_references, best_similarities
    

if __name__ == '__main__':

    text = 'Hi! Of course I can! I\'m not really sure, but I\'ll try my best! A neural network is a computer system modeled after the human brain that is designed to recognize patterns. It is composed of layers of interconnected nodes, called neurons, that process and transmit information. Each connection between neurons has a weight that determines the strength of the connection.'

    mov_rec = Movement_recognition()

    sentences, actions, references, similarities = mov_rec.detect_actions(text)
    for index in range(len(sentences)):
        print('Sentence:', sentences[index])
        print('Action selected:', actions[index])
        print('Because it\'s similar to:', references[index])
        print('Similarity:', similarities[index])
        print()