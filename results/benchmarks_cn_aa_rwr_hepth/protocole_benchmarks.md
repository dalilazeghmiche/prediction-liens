# Protocole des benchmarks (article [2] — hep-th)

- Source : hep-th KDD Cup 2003, années SLAC.
- Apprentissage : 1994–1996 ; cœur historique : >=3 articles en train ET en test 1997–1999.
- Sous-graphes : **induits**, échantillonnage pseudo-aléatoire avec graine 42, sous-ensembles emboîtés.
- Tailles (sommets) : [500, 1000, 2000, 4112] ; RÉPÉTITIONS : 3 par taille et méthode.
- Pour chaque taille : les cinq méthodes utilisent exactement les mêmes sommets,
  arêtes et sommets candidats du cœur restreint.
- Mesures : CN/AA optimisés du dépôt, CN/AA naïfs de ce script (références),
  RWR symétrisé du dépôt ; tous calculent les scores **et** trient le résultat.
- RWR : alpha=0.15, tol=1e-10, max_iter=1000, block_size=64.
- Chronométrage : `time.perf_counter()` dans un processus enfant par méthode et taille,
  avec plusieurs répétitions successives. Bibliothèques BLAS fixées à 1 thread.
  Import des bibliothèques et chargement des données exclus du chronométrage.
- Mémoire : pic RSS du processus enfant, **échantillonné** toutes les ~5 ms
  dans un thread indépendant. Le pic est commun aux répétitions du même processus.
  La colonne supplémentaire est max(0, pic RSS - RSS juste avant calcul).
  Le pic réel peut être plus élevé ; l'exactitude est limitée pour les étapes très rapides.
- Résumés : médiane, minimum et maximum des temps ; médiane des pics RSS observés.
- La génération des CSV/figures n'est pas comprise dans les mesures.
- Les tailles croissantes servent aux BENCHMARKS ALGORITHMIQUES.
  Les sous-graphes ne reproduisent pas l'expérience temporelle de prédiction de liens.
- Les scores RWR sont symétrisés : r_u(v) + r_v(u), adaptation de l'article [2].
- CN/AA naïfs et optimisés : égalité des scores vérifiée sur chaque
  sous-graphe (AA à une tolérance flottante de 1e-12), hors chronométrage.
- Sur de très grands graphes, RWR peut consommer beaucoup de mémoire car la
  fonction du dépôt construit R (n × |coeur|) et S (|coeur| × |coeur|) denses.
- Le corpus de ce dépôt diffère du corpus exact de l'article [2].
