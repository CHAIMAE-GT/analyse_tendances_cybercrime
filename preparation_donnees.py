import pandas as pd
import spacy
import re
from nltk.corpus import stopwords

# Charger les données depuis le fichier CSV généré
df = pd.read_csv('tweets_cleaned.csv')
try:
    df = pd.read_csv('tweets_cleaned.csv')
except FileNotFoundError:
    print("Le fichier 'tweets_cleaned.csv' est introuvable.")
except pd.errors.EmptyDataError:
    print("Le fichier 'tweets_cleaned.csv' est vide.")


# Vérifier les premières lignes et les valeurs manquantes
print(df.head())
print(df.isnull().sum())

# Initialiser spaCy
nlp = spacy.load('en_core_web_sm')

# Fonction pour nettoyer les tweets
def nettoyer_texte(texte):
    # Retirer les URL
    texte = re.sub(r'http\S+', '', texte)
    # Retirer les mentions de comptes
    texte = re.sub(r'@\w+', '', texte)
    # Retirer les hashtags
    texte = re.sub(r'#\w+', '', texte)
    # Tokenisation et suppression des stopwords
    doc = nlp(texte.lower())
    tokens = [token.text for token in doc if token.text not in stopwords.words('english') and token.text.isalpha()]
    return " ".join(tokens)

# Appliquer la fonction de nettoyage à la colonne 'tweet_text'
df['tweet_text'] = df['tweet_text'].apply(nettoyer_texte)

# Convertir 'created_at' en format datetime
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Sauvegarder les données nettoyées dans un nouveau fichier CSV
df.to_csv('donnees_nettoyees.csv', index=False)

print("Nettoyage terminé, données sauvegardées dans 'donnees_nettoyees.csv'.")
