"""Interface graphique (Streamlit) pour explorer les expériences de prédiction de liens.

Lancement, depuis la racine du dépôt :

    python -m pip install -r requirements-interface.txt
    python -m streamlit run interface.py

L'interface est un complément facultatif : elle appelle exactement les mêmes
fonctions que les scripts lancés par run.sh (src/ et utils_experiences.py).
Les résultats officiels du rapport restent ceux produits par run.sh.
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import Counter, defaultdict
from heapq import nsmallest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.sparse.csgraph import connected_components, shortest_path  # noqa: E402

from preparer_donnees_l3 import lire_hi_ii_14  # noqa: E402
from src.datasets import load_hepth  # noqa: E402
from src.graph_io import num_edges  # noqa: E402
from src.measures.global_measures import rwr_scores  # noqa: E402
from src.measures.l3 import l3_optimise  # noqa: E402
from src.measures.local import aa_scores, cn_scores  # noqa: E402
from src.split import pair_key, random_edge_split, temporal_split  # noqa: E402
from utils_experiences import RAW, courbe_attendue  # noqa: E402

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

HEPTH_FILES = (RAW / "hep-th-abs.tar.gz", RAW / "hep-th-slacdates.tar.gz")
HI_FILE = RAW / "HI-II-14.tsv"

JEUX = {
    "hepth": "hep-th : co-auteurs arXiv (réseau social)",
    "hi": "HI-II-14 : interactions entre protéines",
}

# Protocoles utilisés dans le rapport (run.sh).
PROTOCOLE_HEPTH = {"train": (1994, 1996), "test": (1997, 1999), "k_train": 3, "k_test": 3}
PROTOCOLE_HI = {"fraction": 0.5, "seed": 42}
ALPHA_RAPPORT = 0.15
CLE_HEPTH_RAPPORT = ("hepth", (1994, 1996), (1997, 1999), 3, 3)
CLE_HI_RAPPORT = ("hi", 0.5, 42)

ALGOS = {
    "CN": {
        "nom": "Voisins communs (CN)",
        "formule": r"s_{\mathrm{CN}}(x,y)=|\Gamma(x)\cap\Gamma(y)|",
        "idee": "Compte les voisins communs : chemins de longueur 2.",
    },
    "AA": {
        "nom": "Adamic-Adar (AA)",
        "formule": r"s_{\mathrm{AA}}(x,y)=\sum_{w\in\Gamma(x)\cap\Gamma(y)}\frac{1}{\ln d(w)}",
        "idee": "Voisins communs pondérés : un voisin commun rare compte plus.",
    },
    "RWR": {
        "nom": "Marche aléatoire avec redémarrage (RWR)",
        "formule": r"s_{\mathrm{RWR}}(x,y)=r_x(y)+r_y(x)",
        "idee": "Probabilité stationnaire d'une marche qui revient à sa source "
                "avec probabilité α ; version symétrisée.",
    },
    "L3": {
        "nom": "L3 (Kovács et al.)",
        "formule": r"s_{\mathrm{L3}}(x,y)=\sum_{u,v}\frac{a_{xu}\,a_{uv}\,a_{vy}}{\sqrt{d(u)\,d(v)}}",
        "idee": "Chemins de longueur 3, normalisés par les degrés des sommets intermédiaires.",
    },
}

COULEURS = {"CN": "#4C78A8", "AA": "#F58518", "RWR": "#54A24B", "L3": "#E45756",
            "Hasard": "#7F7F7F"}
COULEURS_RESEAUX = {"hep-th · apprentissage": "#4C78A8",
                    "HI-II-14 · complet": "#E45756",
                    "HI-II-14 · apprentissage": "#F4A6A6"}

TRANCHES = ["2", "3", "4", "5", "6 et plus", "non reliées"]

TOP_MIN = 1000  # nombre minimal de prédictions gardées pour l'onglet Prédictions


# --------------------------------------------------------------------------
# Données et découpages (mis en cache : calculés une seule fois)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Lecture des articles hep-th…")
def charger_hepth():
    manquants = [str(p) for p in HEPTH_FILES if not p.is_file()]
    if manquants:
        raise FileNotFoundError("Fichier(s) manquant(s) : " + ", ".join(manquants))
    return load_hepth(*HEPTH_FILES)


@st.cache_resource(show_spinner="Lecture de HI-II-14…")
def charger_hi():
    return lire_hi_ii_14(HI_FILE)


@st.cache_resource(show_spinner="Découpage du réseau…")
def obtenir_split(cle):
    """`cle` décrit le protocole : ("hepth", train, test, k_train, k_test)
    ou ("hi", fraction, seed)."""
    if cle[0] == "hepth":
        _, train, test, k_train, k_test = cle
        return temporal_split(charger_hepth(), train, test, k_train, k_test)
    _, fraction, seed = cle
    adj, _ = charger_hi()
    return random_edge_split(adj, test_fraction=fraction, seed=seed)


@st.cache_resource(show_spinner=False, max_entries=10)
def calculer_scores(cle, algo, alpha):
    """Scores > 0 des paires candidates non reliées, et temps de calcul (s)."""
    split = obtenir_split(cle)
    adj, coeur = split.train_adj, split.core
    debut = time.perf_counter()
    if algo == "CN":
        scores = cn_scores(adj, coeur)
    elif algo == "AA":
        scores = aa_scores(adj, coeur)
    elif algo == "L3":
        scores = l3_optimise(adj, coeur=coeur)
    elif algo == "RWR":
        scores = rwr_scores(adj, nodes=sorted(coeur), alpha=alpha, block_size=64)
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")
    return scores, time.perf_counter() - debut


@st.cache_resource(show_spinner=False, max_entries=10)
def scores_tries(cle, algo, alpha):
    """Scores triés par ordre croissant, pour retrouver le rang d'une paire."""
    scores, _ = calculer_scores(cle, algo, alpha)
    return np.sort(np.fromiter(scores.values(), dtype=float, count=len(scores)))


def rang(tries, score):
    """1 + nombre de paires de score strictement supérieur."""
    return int(len(tries) - np.searchsorted(tries, score, side="right")) + 1


def echantillonner(rows, k_ref):
    """Garde ~1 000 points d'une courbe (tous les premiers, puis géométrique)."""
    n = len(rows)
    idx = set(range(min(n, 200)))
    if n > 200:
        idx.update(int(i) - 1 for i in np.unique(np.geomspace(200, n, 800).astype(int)))
    idx.update({min(k_ref, n) - 1, n - 1})
    return [rows[i] for i in sorted(idx)]


@st.cache_resource(show_spinner=False, max_entries=20)
def evaluer(cle, algo, alpha, k):
    """Indicateurs au seuil k, courbe attendue et meilleures prédictions."""
    scores, duree = calculer_scores(cle, algo, alpha)
    split = obtenir_split(cle)
    vrais, n_cand = split.new_edges, split.num_candidates
    hasard = len(vrais) / n_cand

    k_courbe = min(max(2 * k, 20_000), n_cand)
    rows, liens_score_positif = courbe_attendue(scores, vrais, n_cand, max(k, k_courbe))
    _, p_k, r_k = rows[k - 1]

    # Même règle de classement que src.measures.local.rank : score
    # décroissant puis paire croissante, sans trier tous les scores.
    n_top = min(len(scores), max(k, TOP_MIN))
    top = nsmallest(n_top, scores.items(), key=lambda item: (-item[1], item[0]))
    det = sum(p in vrais for p, _ in top[:k]) if len(scores) >= k else None

    return {
        "algo": algo,
        "k": k,
        "tp_esperance": p_k * k,
        "tp_deterministe": det,
        "precision": p_k,
        "rappel": r_k,
        "facteur": p_k / hasard,
        "liens_score_positif": liens_score_positif,
        "paires_notees": len(scores),
        "temps": duree,
        "courbe": echantillonner(rows, k),
        "top": [(u, v, s, (u, v) in vrais) for (u, v), s in top],
    }


# --------------------------------------------------------------------------
# Analyse de la structure des réseaux
# --------------------------------------------------------------------------

def matrice(adj):
    """Sommets triés, index {sommet: i} et matrice d'adjacence creuse."""
    nodes = sorted(adj)
    index = {u: i for i, u in enumerate(nodes)}
    rows, cols = [], []
    for u in nodes:
        iu = index[u]
        for v in adj[u]:
            rows.append(iu)
            cols.append(index[v])
    n = len(nodes)
    a = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    return nodes, index, a


def mesurer_structure(adj):
    """Indicateurs globaux, distribution des degrés et clustering par degré."""
    nodes, _, a = matrice(adj)
    deg = np.array([len(adj[u]) for u in nodes], dtype=np.int64)
    m = int(deg.sum()) // 2
    n = len(nodes)

    # Triangles : pour chaque arête u-v, chaque voisin commun w ferme un
    # triangle ; on le compte pour w (arête entre deux voisins de w).
    tri = Counter()
    for u in nodes:
        for v in adj[u]:
            if u < v:
                petit, grand = ((adj[u], adj[v]) if len(adj[u]) <= len(adj[v])
                                else (adj[v], adj[u]))
                for w in petit:
                    if w in grand:
                        tri[w] += 1
    triangles = sum(tri.values()) // 3
    paires_voisins = deg * (deg - 1) // 2
    total_paires = int(paires_voisins.sum())
    local = np.array([tri[u] / paires_voisins[i]
                      for i, u in enumerate(nodes) if deg[i] >= 2])
    _, labels = connected_components(a, directed=False)
    tailles = np.bincount(labels)
    n_comp = int((tailles > 1).sum())          # composantes d'au moins une arête
    r, c = a.nonzero()
    assort = (float(np.corrcoef(deg[r], deg[c])[0, 1])
              if len(r) > 1 and deg[r].std() > 0 else float("nan"))

    stats = {
        "Sommets": n,
        "Arêtes": m,
        "Degré moyen": 2 * m / n if n else 0.0,
        "Degré maximal": int(deg.max()) if n else 0,
        "Sommets isolés (degré 0)": int((deg == 0).sum()),
        "Composantes connexes (≥ 2 sommets)": n_comp,
        "Composante géante (% des sommets)": tailles.max() / n if n else 0.0,
        "Triangles": triangles,
        "Transitivité (clustering global)": 3 * triangles / total_paires if total_paires else 0.0,
        "Clustering local moyen": float(local.mean()) if len(local) else 0.0,
        "Assortativité des degrés": assort,
    }
    deg_df = pd.DataFrame(sorted(Counter(int(d) for d in deg if d > 0).items()),
                          columns=["degré", "sommets"])
    par_degre = defaultdict(list)
    for i, u in enumerate(nodes):
        if deg[i] >= 2:
            par_degre[int(deg[i])].append(tri[u] / paires_voisins[i])
    clust_df = pd.DataFrame([(d, float(np.mean(v)), len(v)) for d, v in sorted(par_degre.items())],
                            columns=["degré", "clustering", "sommets"])
    return stats, deg_df, clust_df


@st.cache_resource(show_spinner="Analyse de la structure du réseau…")
def structure_split(cle):
    return mesurer_structure(obtenir_split(cle).train_adj)


@st.cache_resource(show_spinner="Analyse de la structure du réseau…")
def structure_hi_complet():
    return mesurer_structure(charger_hi()[0])


def tranche(distances):
    """Distance -> indice dans TRANCHES (2, 3, 4, 5, 6+, non reliées)."""
    d = np.asarray(distances, dtype=float)
    out = np.full(d.shape, 5, dtype=np.int64)
    fini = np.isfinite(d)
    out[fini] = np.clip(d[fini].astype(np.int64) - 2, 0, 4)
    return out


@st.cache_resource(show_spinner="Calcul des distances dans le graphe d'apprentissage…")
def distances(cle):
    """Distances (graphe d'apprentissage) de toutes les paires candidates et
    des liens à prédire. Parcours en largeur depuis les sommets du cœur, par
    blocs, sans stocker la matrice complète des distances."""
    split = obtenir_split(cle)
    nodes, index, a = matrice(split.train_adj)
    coeur = np.array(sorted(index[u] for u in split.core), dtype=np.int64)
    par_source = defaultdict(list)
    for u, v in split.new_edges:
        i, j = sorted((index[u], index[v]))
        par_source[i].append(j)

    cand = np.zeros(len(TRANCHES), dtype=np.int64)
    dist_test = {}
    for debut in range(0, len(coeur), 256):
        src = coeur[debut:debut + 256]
        d = shortest_path(a, method="D", directed=False, unweighted=True, indices=src)
        vals = d[:, coeur][coeur[None, :] > src[:, None]]   # paires i < j du cœur
        vals = vals[vals != 1]                                # 1 = déjà reliées
        cand += np.bincount(tranche(vals), minlength=len(TRANCHES))
        for r, i in enumerate(src):
            for j in par_source.get(int(i), ()):
                dist_test[(nodes[i], nodes[j])] = float(d[r, j])
    test = np.bincount(tranche(list(dist_test.values())), minlength=len(TRANCHES))
    return cand, test, dist_test


@st.cache_resource(show_spinner=False)
def partenaires_test(cle):
    part = defaultdict(set)
    for u, v in obtenir_split(cle).new_edges:
        part[u].add(v)
        part[v].add(u)
    return dict(part)


# --------------------------------------------------------------------------
# Dessin du voisinage d'un sommet (Graphviz, rendu dans le navigateur)
# --------------------------------------------------------------------------

def _q(texte):
    return '"' + str(texte).replace("\\", "\\\\").replace('"', '\\"') + '"'


def etiquette(nom, bio):
    return "…" + nom[-6:] if bio and len(nom) > 12 else str(nom)


def dot_voisinage(adj, centre, rayon, max_voisins, part, predictions, bio, coeur):
    """Code DOT du voisinage de `centre`.

    - arêtes d'apprentissage en gris ;
    - liens à prédire (entre sommets affichés) en orange pointillé ;
    - prédictions de l'algorithme choisi : vert si correctes, rouge sinon.
    """
    futurs_tous = part.get(centre, set())
    voisins = sorted(adj[centre], key=lambda v: (-len(adj[v]), v))
    garde = set(voisins[:max_voisins])
    # Pour rester lisible : au plus `max_voisins` futurs partenaires (les liens
    # bien prédits d'abord) et `max_voisins` prédictions fausses (les mieux
    # classées d'abord, `predictions` suit l'ordre du classement).
    futurs_tries = sorted(futurs_tous, key=lambda v: (predictions.get(v) is not True,
                                                      -len(adj[v] & adj[centre]),
                                                      -len(adj[v]), v))
    futurs = set(futurs_tries[:max_voisins])
    fausses = [v for v, ok in predictions.items() if not ok]
    fausses_aff = set(fausses[:max_voisins])
    omis = {"voisins": len(voisins) - len(garde),
            "liens à prédire": len(futurs_tous) - len(futurs),
            "prédictions fausses": len(fausses) - len(fausses_aff)}
    predictions = {v: ok for v, ok in predictions.items() if v in futurs or v in fausses_aff}
    noeuds = {centre} | garde | futurs | fausses_aff
    if rayon == 2:
        second = Counter(w for v in garde for w in adj[v] if w not in noeuds)
        noeuds |= {w for w, _ in second.most_common(max_voisins)}

    lignes = [
        "graph G {",
        'layout=neato; overlap=false; splines=true; outputorder=edgesfirst; '
        'bgcolor="transparent"; pad=0.3; sep="+6"; ratio=0.62;',
        'node [shape=ellipse, style=filled, fontname="Helvetica", fontsize=10, '
        'color="#9CA3AF", fillcolor="#F3F4F6", penwidth=1, height=0.3];',
        'edge [color="#C4C4C4", penwidth=1];',
    ]
    for u in sorted(noeuds):
        attrs = {"label": etiquette(u, bio), "tooltip": f"{u} (degré {len(adj[u])})"}
        if u == centre:
            attrs.update(fillcolor="#111827", fontcolor="white", color="#111827", fontsize="12")
        elif u in futurs:
            attrs.update(fillcolor="#FDE7CF", color="#F58518", penwidth="2")
        elif u in predictions:
            attrs.update(fillcolor="#FDE2E1", color="#D62728", penwidth="1.5")
        elif u in garde:
            attrs.update(fillcolor="#DCE7F3", color="#4C78A8")
        if coeur is not None and u not in coeur and u != centre:
            attrs.update(fontcolor="#6B7280", style="filled,dashed")
        lignes.append(f"{_q(u)} [" + ", ".join(f"{k}={_q(v)}" for k, v in attrs.items()) + "];")

    for u in noeuds:
        for v in adj[u]:
            if v in noeuds and u < v:
                lignes.append(f"{_q(u)} -- {_q(v)};")
    for u in noeuds:
        for v in part.get(u, ()):
            if v in noeuds and u < v and (centre not in (u, v) or (v if u == centre else u) in futurs):
                autre = v if u == centre else u if v == centre else None
                if autre is not None and predictions.get(autre) is True:
                    style = 'color="#2CA02C", penwidth=3'
                else:
                    style = 'color="#F58518", style=dashed, penwidth=2'
                lignes.append(f"{_q(u)} -- {_q(v)} [{style}];")
    for autre, correct in predictions.items():
        if not correct:
            lignes.append(f'{_q(centre)} -- {_q(autre)} [color="#D62728", style=dotted, penwidth=2.2];')
    lignes.append("}")
    return "\n".join(lignes), len(noeuds), omis


# --------------------------------------------------------------------------
# Petits utilitaires d'affichage
# --------------------------------------------------------------------------

def fr(x, decimales=0):
    """Nombre au format français : espace fine pour les milliers, virgule."""
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def pct(x, decimales=2):
    return fr(100 * x, decimales) + " %"


def formater_stat(nom, valeur):
    if isinstance(valeur, float) and np.isnan(valeur):
        return "—"
    if nom.startswith(("Composante géante",)):
        return pct(valeur, 1)
    if nom.startswith(("Transitivité", "Clustering")):
        return fr(valeur, 3)
    if nom.startswith("Assortativité"):
        return fr(valeur, 3)
    if nom.startswith("Degré moyen"):
        return fr(valeur, 2)
    return fr(valeur)


def echelle_couleurs(algos):
    noms = list(algos) + ["Hasard"]
    return alt.Scale(domain=noms, range=[COULEURS[a] for a in noms])


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Prédiction de liens - AAGA", layout="wide")
st.title("Prédiction de liens dans les réseaux complexes")
st.caption("Projet AAGA, Sujet 8. Les calculs appellent le code de `src/`, "
           "comme `run.sh` ; les chiffres officiels du rapport restent ceux de `run.sh`.")

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Expérience")
    jeu = st.radio("Jeu de données", list(JEUX), format_func=JEUX.get)

    with st.form("parametres"):
        if jeu == "hepth":
            st.markdown("**Protocole temporel** (Liben-Nowell et Kleinberg)")
            train = st.slider("Période d'apprentissage", 1992, 2003,
                              PROTOCOLE_HEPTH["train"])
            test = st.slider("Période de test", 1992, 2003, PROTOCOLE_HEPTH["test"])
            k_train = st.number_input("κ apprentissage", 1, 10, PROTOCOLE_HEPTH["k_train"],
                                      help="Articles minimum dans la période d'apprentissage "
                                           "pour qu'un auteur soit dans le cœur.")
            k_test = st.number_input("κ test", 1, 10, PROTOCOLE_HEPTH["k_test"],
                                     help="Articles minimum dans la période de test.")
            dispo, defaut = ["CN", "AA", "RWR", "L3"], ["CN", "AA", "RWR"]
        else:
            st.markdown("**Masquage aléatoire** (Kovács et al.)")
            fraction = st.slider("Part des interactions masquées", 0.1, 0.9,
                                 PROTOCOLE_HI["fraction"], 0.05)
            seed = st.number_input("Graine du tirage", 0, 100_000, PROTOCOLE_HI["seed"])
            dispo, defaut = ["CN", "AA", "L3"], ["CN", "AA", "L3"]

        algos = st.multiselect("Algorithmes", dispo, defaut,
                               format_func=lambda a: ALGOS[a]["nom"])
        alpha = ALPHA_RAPPORT
        if jeu == "hepth":
            alpha = st.slider("α de RWR (probabilité de retour)", 0.05, 0.95,
                              ALPHA_RAPPORT, 0.05)
        k_choisi = st.number_input("k : nombre de paires prédites (0 = nombre de liens à prédire)",
                                   0, 1_000_000, 0, step=100)
        lancer = st.form_submit_button("Lancer l'expérience", type="primary")

    if jeu == "hi":
        st.caption("RWR n'est pas proposé sur HI-II-14 : il faudrait une marche par "
                   "protéine et stocker des millions de scores, ce que le projet "
                   "n'évalue pas sur ce réseau.")

# ---------------------------------------------------------------- protocole choisi
if jeu == "hepth":
    if train[1] >= test[0]:
        st.error("La période de test doit commencer après la fin de la période d'apprentissage.")
        st.stop()
    cle = ("hepth", tuple(train), tuple(test), int(k_train), int(k_test))
else:
    cle = ("hi", round(float(fraction), 2), int(seed))

# Les résultats affichés sont ceux du dernier « Lancer » pour CE jeu de données.
if lancer:
    st.session_state[f"config_{jeu}"] = {"cle": cle, "algos": list(algos),
                                         "alpha": float(alpha), "k": int(k_choisi)}
config = st.session_state.get(f"config_{jeu}")
cle_affichee = config["cle"] if config else cle
officiel = cle_affichee in (CLE_HEPTH_RAPPORT, CLE_HI_RAPPORT)
bio = cle_affichee[0] == "hi"

try:
    split = obtenir_split(cle_affichee)
except FileNotFoundError as err:
    st.error(f"{err}\n\nPlacer les données dans `data/raw/` (voir le README).")
    st.stop()

n_liens = len(split.new_edges)
n_cand = split.num_candidates
hasard = n_liens / n_cand if n_cand else 0.0

# ---------------------------------------------------------------- calculs
zone_calcul = st.container()
resultats = []
if config and config["algos"]:
    k = max(1, min(config["k"] or n_liens, n_cand))
    with zone_calcul.status("Calcul des scores…", expanded=False) as statut:
        for algo in config["algos"]:
            statut.update(label=f"Calcul de {ALGOS[algo]['nom']}…")
            a = config["alpha"] if algo == "RWR" else None
            resultats.append(evaluer(config["cle"], algo, a, k))
        statut.update(label=f"Calculs terminés ({len(resultats)} algorithme(s), k = {fr(k)})",
                      state="complete")
noms = [r["algo"] for r in resultats]

onglet_donnees, onglet_resultats, onglet_courbes, onglet_predictions = st.tabs(
    ["Jeu de données", "Résultats", "Courbes", "Prédictions"])

# ================================================================ 1. jeu de données
with onglet_donnees:
    if officiel:
        st.success("Paramètres identiques au protocole du rapport.")
    else:
        st.warning("Paramètres différents du protocole du rapport : résultats exploratoires.")

    s_vue, s_structure, s_liens, s_explorer = st.tabs(
        ["Vue d'ensemble", "Structure du réseau", "Liens à prédire", "Explorer un sommet"])
    adj = split.train_adj

    # ------------------------------------------------------------ vue d'ensemble
    with s_vue:
        if not bio:
            _, tr, te, kt, kte = cle_affichee
            papiers = charger_hepth()
            n_art_train = sum(tr[0] <= y <= tr[1] for y, _ in papiers)
            n_art_test = sum(te[0] <= y <= te[1] for y, _ in papiers)
            st.subheader("Réseau de co-auteurs hep-th (arXiv, KDD Cup 2003)")
            st.markdown(
                f"Un sommet est un auteur ; deux auteurs sont reliés s'ils ont signé un article "
                f"ensemble. Le graphe d'apprentissage couvre **{tr[0]}–{tr[1]}**. Il faut "
                f"prédire les **nouvelles** collaborations de **{te[0]}–{te[1]}** entre les "
                f"auteurs du cœur (au moins {kt} articles en apprentissage et {kte} en test).")
            c = st.columns(4)
            c[0].metric("Articles (apprentissage)", fr(n_art_train))
            c[1].metric("Auteurs", fr(len(adj)))
            c[2].metric("Arêtes d'apprentissage", fr(num_edges(adj)))
            c[3].metric("Auteurs du cœur", fr(len(split.core)))
            c = st.columns(4)
            c[0].metric("Articles (test)", fr(n_art_test))
            c[1].metric("Liens à prédire", fr(n_liens))
            c[2].metric("Paires candidates", fr(n_cand))
            c[3].metric("Précision du hasard", pct(hasard, 3))

            annees = Counter(y for y, _ in papiers if 1992 <= y <= 2003)
            taille_equipes = defaultdict(list)
            for y, auteurs in papiers:
                if 1992 <= y <= 2003:
                    taille_equipes[y].append(len(auteurs))
            df_annees = pd.DataFrame(
                [{"année": y, "articles": n,
                  "auteurs par article": float(np.mean(taille_equipes[y])),
                  "rôle": ("apprentissage" if tr[0] <= y <= tr[1]
                           else "test" if te[0] <= y <= te[1] else "non utilisé")}
                 for y, n in sorted(annees.items())])
            st.altair_chart(
                alt.Chart(df_annees, title="Articles par année")
                .mark_bar()
                .encode(x=alt.X("année:O", title=None),
                        y=alt.Y("articles:Q", title="articles"),
                        color=alt.Color("rôle:N", scale=alt.Scale(
                            domain=["apprentissage", "test", "non utilisé"],
                            range=[COULEURS["CN"], COULEURS["AA"], "#C7C7C7"]),
                            legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                        tooltip=["année", "articles", "rôle",
                                 alt.Tooltip("auteurs par article:Q", format=".2f")])
                .properties(height=280),
                width="stretch")
        else:
            _, frac, graine = cle_affichee
            adj_complet, infos = charger_hi()
            isoles = {u for u in adj if not adj[u]}
            inaccessibles = sum(u in isoles or v in isoles for u, v in split.new_edges)
            st.subheader("Interactome humain HI-II-14 (levure double hybride)")
            st.markdown(
                f"Un sommet est une protéine ; une arête est une interaction binaire mesurée. "
                f"Le réseau n'est pas daté : on masque au hasard **{pct(frac, 0)}** des "
                f"interactions (graine {graine}) et on cherche à les retrouver à partir des autres.")
            c = st.columns(4)
            c[0].metric("Protéines", fr(len(adj_complet)))
            c[1].metric("Interactions (après nettoyage)", fr(num_edges(adj_complet)))
            c[2].metric("Interactions d'apprentissage", fr(num_edges(adj)))
            c[3].metric("Interactions masquées (à retrouver)", fr(n_liens))
            c = st.columns(4)
            c[0].metric("Paires candidates", fr(n_cand))
            c[1].metric("Précision du hasard", pct(hasard, 4))
            c[2].metric("Protéines isolées en apprentissage", fr(len(isoles)))
            c[3].metric("Liens masqués inaccessibles", fr(inaccessibles),
                        help="Liens masqués dont une extrémité n'a plus aucune arête en "
                             "apprentissage : aucune mesure fondée sur les chemins ne peut "
                             "leur donner un score positif.")
            st.caption(f"Nettoyage du fichier : {fr(infos['lignes_donnees'])} lignes, "
                       f"{fr(infos['boucles_ignorees'])} auto-interactions et "
                       f"{fr(infos['doublons_ignores'])} doublons retirés. Une paire absente de "
                       "la base n'est pas une non-interaction prouvée.")

    # ------------------------------------------------------------ structure
    with s_structure:
        st.markdown("Les mesures locales reposent sur les **triangles** (voisins communs) ; "
                    "L3 repose sur les **chemins de longueur 3**. Comparer la structure des "
                    "deux réseaux aide à comprendre quelle mesure convient à chacun.")
        reseaux = {}
        cle_h = cle_affichee if not bio else CLE_HEPTH_RAPPORT
        cle_b = cle_affichee if bio else CLE_HI_RAPPORT
        manquants = []
        try:
            reseaux["hep-th · apprentissage"] = structure_split(cle_h)
        except FileNotFoundError:
            manquants.append("hep-th")
        try:
            reseaux["HI-II-14 · complet"] = structure_hi_complet()
            reseaux["HI-II-14 · apprentissage"] = structure_split(cle_b)
        except FileNotFoundError:
            manquants.append("HI-II-14")
        if manquants:
            st.info("Données absentes, réseau non comparé : " + ", ".join(manquants))

        if reseaux:
            indicateurs = list(next(iter(reseaux.values()))[0])
            tableau = pd.DataFrame(
                {"Indicateur": indicateurs,
                 **{nom: [formater_stat(i, s[0][i]) for i in indicateurs]
                    for nom, s in reseaux.items()}})
            st.dataframe(tableau, hide_index=True, width="stretch")
            st.caption("Transitivité : part des paires de voisins d'un sommet qui sont "
                       "elles-mêmes reliées (3 × triangles / paires de voisins). Clustering "
                       "local : même rapport calculé sommet par sommet, puis moyenné (degré ≥ 2). "
                       "Assortativité : corrélation entre les degrés des deux extrémités d'une arête.")

            if "hep-th · apprentissage" in reseaux and "HI-II-14 · complet" in reseaux:
                t_h = reseaux["hep-th · apprentissage"][0]["Transitivité (clustering global)"]
                t_b = reseaux["HI-II-14 · complet"][0]["Transitivité (clustering global)"]
                if t_b > 0:
                    st.markdown(
                        f"Dans hep-th, **{pct(t_h, 1)}** des paires de voisins d'un auteur sont "
                        f"reliées entre elles, contre **{pct(t_b, 1)}** dans HI-II-14 complet, "
                        f"soit {fr(t_h / t_b, 1)} fois plus de triangles relativement. "
                        "Un article à plusieurs auteurs crée d'un coup un groupe entièrement "
                        "relié, alors que deux protéines qui partagent un partenaire "
                        "interagissent rarement entre elles.")

            log_ccdf = st.toggle("Distribution cumulée (part des sommets de degré ≥ d)",
                                 value=False)
            lignes_deg = []
            for nom, (_, deg_df, _) in reseaux.items():
                total = deg_df["sommets"].sum()
                if log_ccdf:
                    cumul = deg_df["sommets"][::-1].cumsum()[::-1] / total
                    for d, v in zip(deg_df["degré"], cumul):
                        lignes_deg.append({"réseau": nom, "degré": d, "valeur": v})
                else:
                    for d, v in zip(deg_df["degré"], deg_df["sommets"] / total):
                        lignes_deg.append({"réseau": nom, "degré": d, "valeur": v})
            echelle_reseaux = alt.Scale(domain=list(reseaux),
                                        range=[COULEURS_RESEAUX[n] for n in reseaux])
            g_deg = (
                alt.Chart(pd.DataFrame(lignes_deg),
                          title="Distribution des degrés (sommets de degré ≥ 1)")
                .mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=1.5)
                .encode(x=alt.X("degré:Q", scale=alt.Scale(type="log"),
                                title="degré d (échelle log)", axis=alt.Axis(format="~s")),
                        y=alt.Y("valeur:Q", scale=alt.Scale(type="log"),
                                title=("part des sommets de degré ≥ d" if log_ccdf
                                       else "part des sommets de degré d"),
                                axis=alt.Axis(format="~%")),
                        color=alt.Color("réseau:N", scale=echelle_reseaux,
                                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                        tooltip=["réseau", "degré", alt.Tooltip("valeur:Q", format=".3%")])
                .properties(height=330))
            lignes_cl = [{"réseau": nom, "degré": d, "clustering": cval, "sommets": s}
                         for nom, (_, _, cl) in reseaux.items()
                         for d, cval, s in cl.itertuples(index=False)]
            g_cl = (
                alt.Chart(pd.DataFrame(lignes_cl),
                          title="Clustering local moyen selon le degré")
                .mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=1.5)
                .encode(x=alt.X("degré:Q", scale=alt.Scale(type="log"),
                                title="degré d (échelle log)", axis=alt.Axis(format="~s")),
                        y=alt.Y("clustering:Q", title="clustering local moyen"),
                        color=alt.Color("réseau:N", scale=echelle_reseaux,
                                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                        tooltip=["réseau", "degré", "sommets",
                                 alt.Tooltip("clustering:Q", format=".3f")])
                .properties(height=330))
            c1, c2 = st.columns(2)
            c1.altair_chart(g_deg, width="stretch")
            c2.altair_chart(g_cl, width="stretch")

    # ------------------------------------------------------------ liens à prédire
    with s_liens:
        cand_d, test_d, dist_test = distances(cle_affichee)
        st.markdown(
            "Distance, **dans le graphe d'apprentissage**, entre les deux extrémités de chaque "
            "lien à prédire, comparée à celle de toutes les paires candidates. CN et AA ne "
            "notent que les paires à distance 2, L3 seulement celles à distance 2 ou 3, RWR "
            "toutes les paires reliées par un chemin. Les paires « non reliées » (composantes "
            "différentes ou sommet isolé) ne peuvent être trouvées par aucune de ces mesures.")
        if int(cand_d.sum()) != n_cand:
            st.warning(f"Contrôle : {fr(int(cand_d.sum()))} paires classées par distance, "
                       f"{fr(n_cand)} paires candidates attendues.")
        c = st.columns(4)
        c[0].metric("Liens à prédire à distance 2", pct(test_d[0] / n_liens, 1))
        c[1].metric("à distance 3", pct(test_d[1] / n_liens, 1))
        c[2].metric("à distance 4 ou plus", pct(test_d[2:5].sum() / n_liens, 1))
        c[3].metric("non reliés", pct(test_d[5] / n_liens, 1))

        df_d = pd.DataFrame(
            [{"distance": t, "ensemble": "liens à prédire", "part": test_d[i] / n_liens}
             for i, t in enumerate(TRANCHES)]
            + [{"distance": t, "ensemble": "toutes les paires candidates",
                "part": cand_d[i] / n_cand} for i, t in enumerate(TRANCHES)])
        st.altair_chart(
            alt.Chart(df_d, title="Répartition par distance dans le graphe d'apprentissage")
            .mark_bar()
            .encode(x=alt.X("distance:N", sort=TRANCHES, title="distance",
                            axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("ensemble:N", sort=["liens à prédire",
                                                            "toutes les paires candidates"]),
                    y=alt.Y("part:Q", title="part de l'ensemble", axis=alt.Axis(format="%")),
                    color=alt.Color("ensemble:N", scale=alt.Scale(
                        domain=["liens à prédire", "toutes les paires candidates"],
                        range=[COULEURS["AA"], "#9CA3AF"]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                    tooltip=["distance", "ensemble", alt.Tooltip("part:Q", format=".2%")])
            .properties(height=320),
            width="stretch")

        st.dataframe(pd.DataFrame([{
            "Distance": t,
            "Paires candidates": int(cand_d[i]),
            "Liens à prédire": int(test_d[i]),
            "Part des liens à prédire (%)": round(100 * test_d[i] / n_liens, 2),
            "Précision du hasard dans la tranche (%)":
                round(100 * test_d[i] / cand_d[i], 3) if cand_d[i] else None,
            "Facteur / hasard global":
                round((test_d[i] / cand_d[i]) / hasard, 1) if cand_d[i] else None,
        } for i, t in enumerate(TRANCHES)]), hide_index=True, width="stretch")
        st.caption("« Précision du hasard dans la tranche » : précision obtenue en tirant au "
                   "hasard parmi les seules paires à cette distance. Elle montre où se "
                   "concentrent les liens à prédire.")

    # ------------------------------------------------------------ explorer
    with s_explorer:
        part = partenaires_test(cle_affichee)
        options = sorted(split.core, key=lambda u: (-len(part.get(u, ())), -len(adj[u]), u))
        c1, c2, c3, c4 = st.columns([3, 1.2, 1, 2])
        # Par défaut : le premier sommet de taille moyenne (lisible) qui a des liens à prédire.
        defaut_centre = next((i for i, u in enumerate(options)
                              if 5 <= len(adj[u]) <= 30 and len(part.get(u, ())) >= 3), 0)
        centre = c1.selectbox(
            "Sommet", options, index=defaut_centre,
            format_func=lambda u: f"{u} (degré {len(adj[u])}, "
                                  f"{len(part.get(u, ()))} lien(s) à prédire)",
            help="Triés par nombre de liens à prédire, puis par degré. On peut taper un nom. "
                 "Les sommets très connectés donnent un dessin chargé : réduire « Voisins max. ».")
        rayon = c2.radio("Voisinage", [1, 2], horizontal=True,
                         format_func=lambda r: "1 saut" if r == 1 else "2 sauts")
        max_v = c3.number_input("Voisins max.", 5, 100, 25, step=5)
        algo_sup = c4.selectbox("Superposer les prédictions de", ["Aucun"] + noms,
                                format_func=lambda a: a if a == "Aucun" else ALGOS[a]["nom"])
        if not noms:
            c4.caption("Lancer une expérience pour superposer des prédictions.")

        predictions = {}
        res_sup = None
        if algo_sup != "Aucun":
            res_sup = next(r for r in resultats if r["algo"] == algo_sup)
            for u, v, _, ok in res_sup["top"][:res_sup["k"]]:
                if centre in (u, v):
                    predictions[v if u == centre else u] = ok

        dot, n_aff, omis = dot_voisinage(adj, centre, rayon, int(max_v), part, predictions, bio,
                                         None if bio else split.core)
        g1, g2 = st.columns([3, 1])
        with g1:
            # Streamlit ne redessine pas toujours un graphe Graphviz dont le code
            # change : une clé qui dépend du contenu force un nouveau dessin.
            with st.container(key=f"graphe_{hashlib.md5(dot.encode()).hexdigest()[:12]}"):
                st.graphviz_chart(dot, width="stretch")
        with g2:
            st.metric("Degré (apprentissage)", fr(len(adj[centre])))
            st.metric("Liens à prédire", fr(len(part.get(centre, ()))))
            a_dist2 = {w for v in adj[centre] for w in adj[v]} - adj[centre] - {centre}
            st.metric("Sommets à distance 2", fr(len(a_dist2)))
            if res_sup:
                bons = sum(predictions.values())
                st.metric(f"Prédictions correctes de {algo_sup} (top-k)",
                          f"{fr(bons)} / {fr(len(predictions))}")
            st.markdown(
                "**Légende**  \n"
                ":gray[━━] arête d'apprentissage  \n"
                ":orange[╍╍] lien à prédire  \n"
                ":green[━━] lien prédit et réel  \n"
                ":red[┅┅] prédiction fausse  \n"
                "Sommet orange : futur partenaire"
                + ("" if bio else "  \nNom grisé : auteur hors du cœur"))
            non_dessines = ", ".join(f"{fr(n)} {quoi}" for quoi, n in omis.items() if n > 0)
            st.caption(f"{fr(n_aff)} sommets affichés. Survoler un sommet affiche son "
                       "nom complet et son degré."
                       + (f" Non dessinés (augmenter « Voisins max. » ou voir les tableaux) : "
                          f"{non_dessines}." if non_dessines else ""))

        # Liens à prédire du sommet : où chaque algorithme les a-t-il classés ?
        futurs = sorted(part.get(centre, ()))
        if futurs:
            lignes = []
            for p in futurs:
                cle_p = pair_key(centre, p)
                ligne = {"Partenaire futur": p,
                         "Degré": len(adj[p]),
                         "Voisins communs": len(adj[centre] & adj[p]),
                         "Distance": ("non reliés" if not np.isfinite(dist_test.get(cle_p, np.inf))
                                      else str(int(dist_test[cle_p])))}
                for r in resultats:
                    a = config["alpha"] if r["algo"] == "RWR" else None
                    sc, _ = calculer_scores(config["cle"], r["algo"], a)
                    s = sc.get(cle_p)
                    ligne[f"Rang {r['algo']}"] = (rang(scores_tries(config["cle"], r["algo"], a), s)
                                                  if s is not None else None)
                lignes.append(ligne)
            df_futurs = pd.DataFrame(lignes)
            for col in df_futurs.columns:
                if col.startswith("Rang "):
                    df_futurs[col] = df_futurs[col].astype("Int64")
            st.markdown(f"**Liens à prédire de {centre}**")
            st.dataframe(df_futurs, hide_index=True, width="stretch")
            if resultats:
                st.caption(f"Rang : position dans le classement de l'algorithme (1 + nombre de "
                           f"paires de score strictement supérieur). Vide : score nul. Une paire "
                           f"est prédite si son rang est ≤ k = {fr(resultats[0]['k'])}.")
            else:
                st.caption("Lancer une expérience pour voir le rang de ces liens dans chaque "
                           "classement.")
        if res_sup and predictions:
            sc, _ = calculer_scores(config["cle"], algo_sup,
                                    config["alpha"] if algo_sup == "RWR" else None)
            st.markdown(f"**Prédictions de {ALGOS[algo_sup]['nom']} impliquant {centre} (top-k)**")
            st.dataframe(pd.DataFrame([{
                "Partenaire prédit": p,
                "Score": sc[pair_key(centre, p)],
                "Voisins communs": len(adj[centre] & adj[p]),
                "Vrai lien": "✓" if ok else "",
            } for p, ok in sorted(predictions.items(),
                                  key=lambda item: -sc[pair_key(centre, item[0])])]),
                hide_index=True, width="stretch",
                column_config={"Score": st.column_config.NumberColumn(format="%.6g")})

# ================================================================ messages d'attente
if not resultats:
    message = ("Choisir au moins un algorithme puis cliquer sur « Lancer l'expérience »."
               if config else
               "Choisir les algorithmes dans le panneau de gauche, puis cliquer sur "
               "« Lancer l'expérience ».")
    for onglet in (onglet_resultats, onglet_courbes, onglet_predictions):
        onglet.info(message)

# ================================================================ 2. résultats
if resultats:
    k = resultats[0]["k"]

    with onglet_resultats:
        meilleur = max(resultats, key=lambda r: r["precision"])
        st.markdown(
            f"**{ALGOS[meilleur['algo']]['nom']}** est la meilleure mesure au seuil "
            f"k = {fr(k)} : précision de **{pct(meilleur['precision'])}**, soit "
            f"**{fr(meilleur['facteur'], 1)} fois** le hasard "
            f"({fr(meilleur['tp_esperance'], 1)} liens retrouvés sur {fr(n_liens)}).")

        tableau = pd.DataFrame([{
            "Algorithme": ALGOS[r["algo"]]["nom"],
            "Liens retrouvés (espérance)": round(r["tp_esperance"], 2),
            "Top-k déterministe": r["tp_deterministe"],
            "Précision@k (%)": round(100 * r["precision"], 2),
            "Rappel@k (%)": round(100 * r["rappel"], 2),
            "Facteur / hasard": round(r["facteur"], 1),
            "Liens à prédire avec score > 0": r["liens_score_positif"],
            "Paires notées (score > 0)": r["paires_notees"],
            "Temps de calcul (s)": round(r["temps"], 3),
        } for r in resultats])
        st.dataframe(tableau, hide_index=True, width="stretch")
        st.caption("« Espérance » : les ex æquo au seuil, y compris les paires de score nul, "
                   "sont départagés uniformément au hasard (comme dans le rapport). "
                   "« Top-k déterministe » : ex æquo départagés par ordre des noms ; vide si "
                   "moins de k paires ont un score positif.")

        df_barres = pd.DataFrame({"algorithme": noms,
                                  "précision (%)": [100 * r["precision"] for r in resultats]})
        barres = (
            alt.Chart(df_barres, title=f"Précision au seuil k = {fr(k)}")
            .mark_bar()
            .encode(y=alt.Y("algorithme:N", sort="-x", title=None),
                    x=alt.X("précision (%):Q",
                            title="précision au seuil k (%) ; pointillés : hasard"),
                    color=alt.Color("algorithme:N", scale=echelle_couleurs(noms), legend=None),
                    tooltip=["algorithme", alt.Tooltip("précision (%):Q", format=".2f")])
        )
        regle = (alt.Chart(pd.DataFrame({"précision (%)": [100 * hasard]}))
                 .mark_rule(color=COULEURS["Hasard"], strokeDash=[4, 4])
                 .encode(x="précision (%):Q",
                         tooltip=[alt.Tooltip("précision (%):Q", title="hasard",
                                              format=".3f")]))
        st.altair_chart((barres + regle).properties(height=60 + 45 * len(noms)),
                        width="stretch")

        with st.expander("Définition des mesures"):
            for a in noms:
                st.markdown(f"**{ALGOS[a]['nom']}** : {ALGOS[a]['idee']}")
                st.latex(ALGOS[a]["formule"])

    # ------------------------------------------------------------ 3. courbes
    with onglet_courbes:
        log_k = st.toggle("Axe k en échelle logarithmique", value=True)
        lignes = [{"algorithme": r["algo"], "k": kk, "précision": p, "rappel": rr}
                  for r in resultats for kk, p, rr in r["courbe"]]
        df_c = pd.DataFrame(lignes)
        echelle_k = alt.Scale(type="log") if log_k else alt.Scale(type="linear")
        couleurs = echelle_couleurs(noms)
        ligne_hasard = (alt.Chart(pd.DataFrame({"algorithme": ["Hasard"], "précision": [hasard]}))
                        .mark_rule(strokeDash=[4, 4])
                        .encode(y="précision:Q",
                                color=alt.Color("algorithme:N", scale=couleurs)))

        pr = (alt.Chart(df_c, title="Précision en fonction du rappel")
              .mark_line()
              .encode(x=alt.X("rappel:Q", axis=alt.Axis(format="%")),
                      y=alt.Y("précision:Q", axis=alt.Axis(format="%")),
                      color=alt.Color("algorithme:N", scale=couleurs,
                                      legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                      tooltip=["algorithme", "k",
                               alt.Tooltip("précision:Q", format=".2%"),
                               alt.Tooltip("rappel:Q", format=".2%")]))
        pk = (alt.Chart(df_c, title="Précision en fonction de k")
              .mark_line()
              .encode(x=alt.X("k:Q", scale=echelle_k, title="k (paires prédites)",
                              axis=alt.Axis(format="~s")),
                      y=alt.Y("précision:Q", axis=alt.Axis(format="%")),
                      color=alt.Color("algorithme:N", scale=couleurs,
                                      legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                      tooltip=["algorithme", "k",
                               alt.Tooltip("précision:Q", format=".2%"),
                               alt.Tooltip("rappel:Q", format=".2%")]))
        seuil = (alt.Chart(pd.DataFrame({"k": [k]}))
                 .mark_rule(color="black", strokeDash=[2, 3]).encode(x="k:Q"))
        c1, c2 = st.columns(2)
        c1.altair_chart((pr + ligne_hasard).properties(height=380).interactive(),
                        width="stretch")
        c2.altair_chart((pk + ligne_hasard + seuil).properties(height=380).interactive(),
                        width="stretch")
        st.caption(f"Courbes en espérance (ex æquo départagés au hasard), jusqu'à "
                   f"k = {fr(max(row[0] for r in resultats for row in r['courbe']))}. "
                   f"Pointillés : précision du hasard ({pct(hasard, 3)}) et seuil k = {fr(k)}.")

    # ------------------------------------------------------------ 4. prédictions
    with onglet_predictions:
        c1, c2, c3 = st.columns([2, 2, 1])
        choix = c1.selectbox("Algorithme", noms, format_func=lambda a: ALGOS[a]["nom"])
        res = next(r for r in resultats if r["algo"] == choix)
        n_max = len(res["top"])
        if n_max == 0:
            st.info("Aucune paire n'a un score positif.")
        else:
            n_aff = c2.slider("Nombre de paires affichées", 1, n_max, min(50, n_max))
            seulement_vrais = c3.checkbox("Seulement les vrais liens")
            df_top = pd.DataFrame(
                [{"Rang": i, "Sommet x": u, "Sommet y": v, "Score": s,
                  "Voisins communs": len(adj[u] & adj[v]),
                  "Vrai lien": "✓" if ok else ""}
                 for i, (u, v, s, ok) in enumerate(res["top"][:n_aff], 1)])
            n_vrais = int((df_top["Vrai lien"] == "✓").sum())
            st.markdown(f"Parmi les **{fr(n_aff)}** premières paires prédites par "
                        f"{ALGOS[choix]['nom']}, **{fr(n_vrais)}** sont de vrais liens "
                        f"({pct(n_vrais / n_aff, 1)}).")
            if seulement_vrais:
                df_top = df_top[df_top["Vrai lien"] == "✓"]
            st.dataframe(
                df_top, hide_index=True, width="stretch",
                column_config={"Score": st.column_config.NumberColumn(format="%.6g")})
            st.caption("Classement déterministe : score décroissant, puis ordre des noms. "
                       "« Vrai lien » : la paire fait partie des liens à prédire (collaboration "
                       "nouvelle pour hep-th, interaction masquée pour HI-II-14). Pour voir une "
                       "paire dans son contexte : Jeu de données > Explorer un sommet.")
