""" Chargement dex jeux de données réels.

1. arXiv hep-th (KDD Cup 2003): réseau de co-auteurs daté, pout le
protocole de [2]. Deux archives, lues directement sans sans les décompresser:
- hep-th-abs.tar.gz   : un fichier .abs par article, avec la ligne << Authors : A,B and C>> (parfois sur plusiers
lignes, avec des affiliations entre prenthéses);
- hep-th-slacdates.tar.gz : << identifiant date >> (ex. << 9501001 1995-01-02>>).
Résultat: liste d'articles (année, [auteurs]) pour split.temporal_split.


2. Réseaux de protéines(dépot Github de [1]) / simples listes d'atetes, 
lues avec graph_io.read_edge_list."""
import os
import re
import tarfile
 
# ---------------------------------------------------------------- noms
 
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\s*|\\.")          # \v c, \"o, \'e ...
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
 
 
def _strip_parentheses(s):
    """Supprime le texte entre parenthèses (affiliations), même imbriqué."""
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)
 
 
def split_authors(field):
    """« A. Smith, B. Jones and C. Doe (MIT) » → ['A. Smith', 'B. Jones', 'C. Doe']."""
    s = _strip_parentheses(" ".join(field.split()))
    s = re.sub(r"\bet al\.?", "", s)
    parts = re.split(r",|;|&|\band\b", s)
    return [p.strip(" .") for p in parts if p.strip(" .")]
 
 
def normalize_author(name):
    """Clé d'auteur : initiale du prénom + nom de famille, en minuscules.
 
    « John Smith », « J. Smith » et « J.Smith » donnent tous « j smith ».
    Les données de 2003 ne sont pas nettoyées : cette normalisation fusionne
    les variantes d'écriture, au prix de rares fusions de deux homonymes
    (même initiale et même nom). À signaler dans le rapport.
    """
    s = _LATEX_CMD.sub("", name)
    s = re.sub(r"[{}~`'\"^]", "", s).replace(".", " ").replace("-", " - ")
    tokens = [t for t in s.lower().split()
              if t.rstrip(",") not in _SUFFIXES]
    tokens = [t for t in tokens if t != "-"]
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0][0]} {tokens[-1]}"
 
 
# ---------------------------------------------------------------- hep-th
 
_AUTHORS = re.compile(r"^Authors?:[ \t]*(.*(?:\n[ \t]+.*)*)", re.M)
 
 
def parse_abs(text):
    """Extrait la liste des auteurs (normalisés, sans doublon) d'un .abs."""
    m = _AUTHORS.search(text)
    if not m:
        return []
    keys = (normalize_author(a) for a in split_authors(m.group(1)))
    return list(dict.fromkeys(k for k in keys if k))
 
 
def _read_slacdates(path):
    """{identifiant: année} depuis l'archive (ou le fichier) des dates."""
    def lines():
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tar:
                for member in tar:
                    if member.isfile():
                        f = tar.extractfile(member)
                        yield from f.read().decode("latin-1").splitlines()
        else:
            with open(path, encoding="latin-1") as f:
                yield from f
    years = {}
    for line in lines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[1]) >= 4 and parts[1][:4].isdigit():
            years[parts[0]] = int(parts[1][:4])
    return years
 
 
def load_hepth(abs_archive, dates_archive):
    """Retourne la liste des articles (année, [auteurs]) de hep-th.
 
    L'année vient du fichier SLAC ; à défaut, du dossier de l'article.
    """
    years = _read_slacdates(dates_archive)
    papers = []
    with tarfile.open(abs_archive) as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".abs")):
                continue
            folder, fname = os.path.split(member.name)
            paper_id = fname[:-4]
            year = years.get(paper_id)
            if year is None:
                folder_year = os.path.basename(folder)
                if not folder_year.isdigit():
                    continue
                year = int(folder_year)
            text = tar.extractfile(member).read().decode("latin-1")
            authors = parse_abs(text)
            if authors:
                papers.append((year, authors))
    return papers
 