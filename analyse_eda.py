import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Charger les données nettoyées
df = pd.read_csv('donnees_nettoyees.csv')

# Affichage des premières lignes du dataset
print(df.head())

# Calcul des fréquences des mots
all_words = ' '.join(df['tweet_text']).split()
word_freq = Counter(all_words)

# Affichage des 10 mots les plus fréquents
print(word_freq.most_common(10))

# Visualisation des fréquences des mots
plt.figure(figsize=(10, 6))
sns.barplot(x=[item[0] for item in word_freq.most_common(10)],
            y=[item[1] for item in word_freq.most_common(10)])
plt.title('Top 10 des mots les plus fréquents')
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.savefig('eda_graphique.png')

# Générer un nuage de mots
wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['tweet_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('eda_wordcloud.png')

# Visualisation de la longueur des tweets
df['tweet_length'] = df['tweet_text'].apply(len)
plt.figure(figsize=(10, 5))
plt.hist(df['tweet_length'], bins=50, color='skyblue')
plt.title('Distribution de la longueur des tweets')
plt.xlabel('Longueur du tweet')
plt.ylabel('Fréquence')
plt.savefig('eda_length_distribution.png')
