import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from bdd import CLIENT_ID,CLIENT_SECRET
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import davies_bouldin_score


client_id = CLIENT_ID
client_secret = CLIENT_SECRET

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

#CLEAN RAW DATABASE
def createDB(origin):
    df_raw= pd.read_csv(origin, sep=',')
    df = df_raw[df_raw['track_genre'] == 'hip-hop'].copy()
    df = df[['track_id','track_name','artists','popularity','acousticness','energy','speechiness','danceability','duration_ms','liveness','tempo','valence']]
    df['release_date'] = pd.NaT

    print(df.shape[0],df.shape[1])
    df.to_csv("second_spotify_cleaned.csv", index=False)


##Method to add the date by batches to the database (used once)
def fetch_release_dates(df, batch_size=50):

    missing = df[df['release_date'].isna()]
    track_ids = missing['track_id'].tolist()

    for i in tqdm(range(0, len(track_ids), batch_size), desc="Fetching release dates"):
        batch_ids = track_ids[i:i+batch_size]
        try:
            results = sp.tracks(batch_ids)['tracks']
            for track in results:
                release_date = track['album']['release_date']
                
                df.loc[df['track_id'] == track['id'], 'release_date'] = release_date
        except Exception as e:
            print(f"Error fetching batch {i}-{i+batch_size}: {e}")

    
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')


#####MISE EN PLACE UTILISATION DB

df = pd.read_csv("final_db.csv",sep=',')
#fetch_release_dates(df)
#df.to_csv("final_db.csv")


df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])
df = df[df['release_date'].dt.year.between(1994, 2022)]

# Select features
features = ['energy', 'valence', 'tempo', 'danceability', 'acousticness','duration_ms']
X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




def elbow():
    #ELBOW METHOD GIVES 5
    wcss = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 20), wcss, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Elbow Method')
    plt.show()


def silhouette():
    #SILHOUETTE GIVES 5
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}, silhouette={score:.3f}")

    

        #other metrics

        

    dbi = davies_bouldin_score(X_scaled, kmeans.labels_)
    print(f"Davies-Bouldin Index: {dbi:.3f}")

    from sklearn.metrics import calinski_harabasz_score

    ch = calinski_harabasz_score(X_scaled, kmeans.labels_)
    print(f"Calinski-Harabasz Index: {ch:.3f}")



def main_method():
    k_optimal = 5
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    

    X = X_scaled
    clusters = kmeans.predict(X)
    df['cluster'] = kmeans.predict(X)
    df['originality'] = np.nan


    df_sorted = df.sort_values('release_date').copy()
    sorted_positions = df_sorted.index.to_list() 
    position_map = {idx: pos for pos, idx in enumerate(sorted_positions)}

    for idx, row in df_sorted.iterrows():
        cluster_id = row['cluster']
        release_date = row['release_date']
        
        prev_tracks = df_sorted[
            (df_sorted['cluster'] == cluster_id) &
            (df_sorted['release_date'] < release_date)
        ]
        
        if prev_tracks.empty:
            originality_score = np.inf
        else:
            prev_positions = [position_map[i] for i in prev_tracks.index]
            prev_features = X[prev_positions]
            centroid = prev_features.mean(axis=0)
            originality_score = np.linalg.norm(X[position_map[idx]] - centroid)
        
        df.at[idx, 'originality'] = originality_score



    df_filtered=df

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    df_filtered = df[(df['release_date'].dt.year >= 1994) & (df['release_date'].dt.year <= 2022)]


    df_filtered['year_month'] = df_filtered['release_date'].dt.to_period('M')

    monthly_stats = df_filtered.groupby('year_month')['originality'].agg(['min', 'max', 'mean']).reset_index()

    plt.figure(figsize=(15,6))

    for _, row in monthly_stats.iterrows():
        x = row['year_month'].to_timestamp()
        plt.vlines(x, row['min'], row['max'], color='blue', alpha=0.7)
        plt.plot(x, row['mean'], 'ro')

    plt.xlabel("Temps")
    plt.ylabel("Originalité")
    plt.title("Y a t-il une uniformisation du rap ?")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()






def print_number_per_year():
    df = pd.read_csv("final_db.csv") 
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    year_counts = df['year'].value_counts().sort_index()
    print("Nombre de morceaux par an:\n")
    for year, count in year_counts.items():
        print(f"{year}: {count}")

