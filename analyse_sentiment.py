from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données nettoyées
df = pd.read_csv('donnees_nettoyees.csv')

# Charger le modèle de classification de sentiment
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyser le sentiment de chaque tweet
sentiments = df['tweet_text'].apply(lambda tweet: sentiment_analyzer(tweet)[0])

# Ajouter les résultats au dataframe
df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
df['sentiment_score'] = sentiments.apply(lambda x: x['score'])

# Affichage des résultats
print(df[['tweet_text', 'sentiment_label', 'sentiment_score']].head())

# Visualisation des sentiments
sns.countplot(data=df, x='sentiment_label', palette='Set2')
plt.title('Répartition des sentiments des tweets')
plt.show()

# Evolution des sentiments au fil du temps
df['created_at'] = pd.to_datetime(df['created_at'])
df.set_index('created_at', inplace=True)
sentiment_over_time = df.resample('D')['sentiment_label'].value_counts().unstack().fillna(0)
sentiment_over_time.plot(kind='line', figsize=(10, 6), title='Evolution des sentiments au fil du temps')
plt.ylabel('Nombre de tweets')
plt.show()
