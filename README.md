# MLprojectINPT


Ce projet a pour but de répondre à la question :

Le rap est-il de moins en moins créatf ?

Il contient une base de donnée final_db.csv créée de mes soins regroupant beaucoup de morceaux de Hip-Hop de 1994 à nos jours.
Il contient aussi un script disposant des méthodes suivantes : 
  createDB(origin) : de la base de donnée d'origine on ne garde que les éléments qui nous intéressent.
  fetch_release_dates(db, size) : on récupère les dates de sortie de tous les morceaux de la base par paquets de taille déterminée.
  elbow : renvoie le graphe selon la elbow method , l'inertie selon le nombre de clusters.
  silhouette : même chose avec la méthode des silhouettes.
  main_method : Elle calcule le score d'originalité de chaque morceau et trace le graphique d'analyse de l'évolution de la réativité au cours du temps.
  print_number_per_year : renvoie le nombre de morceaux sortis chaque année.
