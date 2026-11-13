# Comparaison RWR directionnel / symétrisé

- Données : hep-th KDD Cup du dépôt, dates SLAC.
- Apprentissage 1994–1996, test 1997–1999, cœur 3 articles par période.
- 1022 auteurs cœur, 520093 candidats, 1164 nouvelles relations à retrouver.
- RWR : alpha=0.15, tol=1e-10, max_iter=1000, block_size=64.
- **Une seule exécution de `rwr_matrix`** produit la même matrice de probabilités pour toutes les variantes.
- `Original_u_vers_v` : r_u(v), où u < v selon l'ordre lexical des identifiants d'auteurs.
- `Original_v_vers_u` : r_v(u), inverse lexical ; les deux sont des formes directionnelles de la définition de l'article.
- `RWR_symetrise` : r_u(v) + r_v(u), la formule du dépôt.
- Le choix lexical est arbitraire et n'est PAS une convention imposée par l'article.
  Ne pas décrire le score `Original_u_vers_v` comme une reproduction littérale complète
  de l'évaluation historique sur des paires non orientées.
- Toutes les variantes utilisent exactement les mêmes paires candidates et les mêmes liens de test.
- En cas d'ex æquo, le Top-k déterministe utilise l'ordre lexical ; la métrique
  en espérance correspond à un départage uniforme au hasard dans chaque groupe.
- Les paires de score nul restent dans les candidats. Les courbes ici s'arrêtent à k=5000.
- Les temps de calcul RWR ne doivent PAS être additionnés ni comparés entre variantes :
  la même matrice a été calculée une seule fois. Seuls les temps de tri diffèrent.
- Si les métriques changent selon l'ordre, cela révèle une sensibilité du RWR directionnel
  à la convention choisie pour les paires non orientées.
- Corpus différent de celui de l'article original, comparaisons internes seulement.
