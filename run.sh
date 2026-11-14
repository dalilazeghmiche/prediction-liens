#!/usr/bin/env bash
# AAGA : expériences des deux parties, depuis la racine du dépôt.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MPLBACKEND=Agg PYTHONHASHSEED=0

help() {
  cat <<'EOF'
Usage : bash run.sh [--all|--social|--bio|--tests|--quick] [--repeats N] [--sizes S]
  --all        Tests + deux parties (défaut)
  --social     hep-th : CN, AA, RWR, courbes, benchmarks
  --bio        HI-II-14 : préparation, L3, longueurs, robustesse, benchmarks
  --tests      Tests unitaires
  --quick      Tests + contrôle hep-th + préparation HI-II-14
  --repeats N  Répétitions des benchmarks (défaut : 3)
  --sizes S    Tailles des sous-graphes : 500,1000,2000,all (défaut)
  --help       Aide
EOF
}

MODE=all; REPEATS=3; SIZES='500,1000,2000,all'; MODE_SEEN=0
while (($#)); do
  case "$1" in
    --all|--social|--bio|--tests|--quick)
      ((MODE_SEEN == 0)) || { echo 'Choisir un seul mode.' >&2; exit 2; }
      MODE="${1#--}"; MODE_SEEN=1; shift ;;
    --repeats)
      [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo '--repeats : entier positif attendu.' >&2; exit 2; }
      REPEATS="$2"; shift 2 ;;
    --sizes)
      [[ -n "${2:-}" ]] || { echo '--sizes : valeur manquante.' >&2; exit 2; }
      SIZES="$2"; shift 2 ;;
    -h|--help) help; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; help >&2; exit 2 ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -f .venv/Scripts/python.exe ]]; then
  PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null; then
  PY=python3
else
  PY=python
fi

run() { "$PY" "$@"; }
tests() { "$PY" -m unittest discover -s tests -q; }
social() {
  run verifier_protocole_partie1.py
  run audit_hepth_article2.py
  run evaluation_cn_aa_hepth.py
  run evaluation_rwr_hepth.py
  run courbes_precision_rappel_hepth.py
  run comparaison_rwr_original_symetrise_hepth.py
  run benchmarks_cn_aa_rwr_hepth.py --repeats "$REPEATS" --sizes "$SIZES"
  run verifier_resultats_social.py
}
bio() {
  run preparer_donnees_l3.py --fraction 0.5 --seed 42
  run evaluer_l3_bio.py
  run benchmarks_l3_bio.py --repeats "$REPEATS" --sizes "$SIZES"
  run comparer_l2_l3_l4_bio.py --max-nodes all
  run robustesse_l3_bio.py --seeds 42,43,44,45,46
}

case "$MODE" in
  all) tests; social; bio ;;
  social) social ;;
  bio) bio ;;
  tests) tests ;;
  quick) tests; run verifier_protocole_partie1.py; run preparer_donnees_l3.py ;;
esac
printf '\n[OK] Campagne terminée. Résultats : results/\n'
