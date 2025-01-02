# Mise à jour du fichier de clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données nettoyées
df = pd.read_csv('donnees_nettoyees.csv')

# Vectorisation des tweets avec TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet_text'])

# Appliquer K-Means pour le clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Ajouter les labels des clusters au dataframe
df['cluster'] = kmeans.labels_

# Affichage des résultats
print(df.head())

# Affichage des mots-clés pour chaque cluster
terms = vectorizer.get_feature_names_out()
for i in range(kmeans.n_clusters):
    print(f"\nCluster {i}:")
    cluster_terms = [terms[index] for index in kmeans.cluster_centers_.argsort()[:, ::-1][i, :10]]
    print("Top mots-clés:", cluster_terms)

# Générer un nuage de mots pour chaque cluster
for i in range(kmeans.n_clusters):
    cluster_texts = " ".join(df[df['cluster'] == i]['tweet_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Nuage de mots pour Cluster {i}")
    plt.axis('off')
    plt.savefig(f'wordcloud_cluster_{i}.png')

# Appliquer PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Visualiser les clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='Set1', style=df['cluster'])
plt.title("Visualisation des clusters K-Means avec PCA")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.savefig('clustering_visualisation_updated.png')
