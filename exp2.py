import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import hmm

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('treebank')

text = input("Enter a sentence: ")

tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)

train_data = nltk.corpus.treebank.tagged_sents()

trainer = hmm.HiddenMarkovModelTrainer()

hmm_tagger = trainer.train(train_data)

hmm_tags = hmm_tagger.tag(tokens)

print("\nHMM POS Tags:")
for word, tag in hmm_tags:
    print(word, "->", tag)

print("\nTag Meanings:")
print("NN  -> Noun")
print("VB  -> Verb")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("PRP -> Pronoun")
print("DT  -> Determiner")

print("\nTotal Words:", len(tokens))
