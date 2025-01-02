import spacy
import nltk
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS

# Télécharger les ressources nécessaires pour NLTK
nltk.download('stopwords')

# Charger le modèle de spaCy pour la langue anglaise ou française
nlp = spacy.load('en_core_web_sm')

# Exemple de texte (tweet)
text = "Caution! Margex is scamming users by charging withdrawal fees..."

# Tokenisation et lemmatisation avec spaCy
doc = nlp(text)
lemmatized_tokens = [token.lemma_ for token in doc if token.text.lower() not in STOP_WORDS]

# Afficher les mots lemmatisés
print("Lemmatisation (spaCy) :", lemmatized_tokens)

# Tokenisation avec NLTK
tokens = nltk.word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

print("Tokenisation (NLTK) :", filtered_tokens)
