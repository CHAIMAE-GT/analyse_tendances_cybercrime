import time
import requests
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer le Bearer Token depuis les variables d'environnement
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
if not BEARER_TOKEN:
    raise ValueError("BEARER_TOKEN est manquant. Veuillez le définir dans le fichier .env.")

# Définir les en-têtes pour l'authentification
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# Fonction pour récupérer les tweets avec logique de retry et pagination
def scrape_tweets_v2(keyword, max_results=10, retries=3, delay=15, pages=1):
    """
    Récupérer les tweets en utilisant l'API de Twitter avec logique de retry et pagination optionnelle.

    Args:
        keyword (str): Mot-clé de recherche.
        max_results (int): Nombre de tweets à récupérer par requête (max 100).
        retries (int): Nombre de tentatives de réessai en cas d'erreurs temporaires.
        delay (int): Délai par défaut en secondes entre les tentatives.
        pages (int): Nombre de pages à récupérer pour la pagination.

    Returns:
        list: Une liste de dictionnaires contenant les données des tweets.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    tweet_data = []
    next_token = None

    for page in range(pages):
        params = {
            "query": keyword,
            "max_results": max_results,
            "tweet.fields": "created_at,text,author_id",
            "user.fields": "username,location",
            "expansions": "author_id"
        }
        if next_token:
            params["next_token"] = next_token

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:  # Requête réussie
                    data = response.json()
                    tweets = data.get('data', [])
                    users = {user['id']: user for user in data.get('includes', {}).get('users', [])}

                    for tweet in tweets:
                        user_info = users.get(tweet['author_id'], {})
                        tweet_info = {
                            'tweet_text': tweet.get('text', 'N/A'),
                            'created_at': tweet.get('created_at', 'N/A'),
                            'username': user_info.get('username', 'Unknown'),
                            'location': user_info.get('location', 'Unknown')
                        }
                        tweet_data.append(tweet_info)

                    # Gérer la pagination
                    next_token = data.get('meta', {}).get('next_token')
                    break

                elif response.status_code == 429:  # Limite de requêtes atteinte
                    reset_time = int(response.headers.get("x-rate-limit-reset", time.time() + delay))
                    sleep_time = reset_time - time.time()
                    print(f"Limite de requêtes atteinte. Nouvelle tentative dans {int(sleep_time)} secondes...")
                    time.sleep(max(sleep_time, delay))

                else:  # Autres erreurs
                    print(f"Erreur : {response.status_code} - {response.json()}")
                    return tweet_data

            except Exception as e:
                print(f"Erreur lors de la récupération des tweets : {e}")
                time.sleep(delay)

        if not next_token:
            break  # Sortir si aucune pagination disponible

    return tweet_data

# Exemple d'utilisation
if __name__ == '__main__':
    keyword = "cybercrime"  # Changez ce mot-clé selon vos besoins
    tweets = scrape_tweets_v2(keyword, max_results=10, retries=5, delay=30, pages=1)  # Récupérer jusqu'à 1 page
    
    if tweets:
        print(f"{len(tweets)} tweets récupérés.")
        for tweet in tweets:
            print(f"{tweet['username']} ({tweet['location']}): {tweet['tweet_text']}\n")
    else:
        print("Aucun tweet trouvé.")
