# 📚 Moteur de recommandation de livres avec K‑Nearest Neighbors

> **Une cellule à la fois** : des CSV bruts jusqu’à un système de recommandation qui passe le test freeCodeCamp.

---

## 🚀 Démarrage rapide

```bash
# 1. Clone / ouvre le notebook Colab fourni par freeCodeCamp
# 2. Exécute les trois premières cellules (téléchargement + chargement des données)
# 3. Copie‑colle le code de la section « 🔧 Pipeline complet » ci‑dessous
# 4. Lance la cellule de test finale → tu devrais voir 🎉🎉🎉
```

---

## 🗺️ Vue d’ensemble du pipeline

```mermaid
flowchart LR
    A[Fichiers CSV bruts] --> B[DataFrames pandas]
    B --> C{Filtres<br/>≥ 200 notes / utilisateur<br/>≥ 100 notes / livre}
    C --> D[Pivot isbn × user]
    D --> E[Matrice creuse CSR]
    E --> F[K‑NN (cosinus)]
    F --> G[get_recommends()]
    G --> H[Top‑5 livres similaires]
```

* **A–B** : `pd.read_csv` avec encodage & séparateur personnalisés.
* **C** : Seuils durs pour retirer le bruit et la haute sparsité.
* **D** : Matrice large où chaque ligne est un vecteur‑livre.
* **E** : Stockage économe (seuls les indices non nuls sont conservés).
* **F** : `NearestNeighbors(metric='cosine', algorithm='brute')`.
* **G** : Wrapper utilitaire qui fait correspondre titres ↔ ISBN et formate la sortie.

---

## 📂 Jeu de données

| Fichier               | Rôle                                | Champs chargés            |
| --------------------- | ----------------------------------- | ------------------------- |
| `BX-Books.csv`        | Métadonnées (\~270 k livres)        | `isbn`, `title`, `author` |
| `BX-Book-Ratings.csv` | 1,1 M notes explicites & implicites | `user`, `isbn`, `rating`  |

*Encodage* : `ISO‑8859‑1`   *Séparateur* : `;`

### Sémantique des notes

* `rating > 0`  →  note explicite (1–10)
* `rating == 0` →  feedback implicite « je possède / j’ai feuilleté »   ✅ conservé dans la matrice

---

## 🔧 Pipeline complet (copier‑coller)

```python
# 0) Imports -----------------------------------------------------------------
import pandas as pd, numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 1) Chargement des CSV ------------------------------------------------------
books = pd.read_csv('BX-Books.csv', encoding='ISO-8859-1', sep=';',
                    usecols=['isbn', 'title', 'author'])
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=';',
                      usecols=['user', 'isbn', 'rating'])

# 2) Filtrage « significatif » ---------------------------------------------
user_good = ratings['user'].value_counts()[lambda s: s>=200].index
book_good = ratings['isbn'].value_counts()[lambda s: s>=100].index
ratings_f = ratings[ratings['user'].isin(user_good) & ratings['isbn'].isin(book_good)]

# 3) Pivot -> matrice livre × utilisateur ----------------------------------
mat_df = ratings_f.pivot(index='isbn', columns='user', values='rating').fillna(0)
mat_csr = csr_matrix(mat_df.values)

# 4) Entraînement K‑NN ------------------------------------------------------
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(mat_csr)

# 5) Dictionnaires d’aide ----------------------------------------------------
isbn2title = books.set_index('isbn')['title']

# 6) Fonction de recommandation --------------------------------------------

def get_recommends(title: str, k: int = 5):
    try:
        isbn = books.loc[books['title'] == title, 'isbn'].iloc[0]
    except IndexError:
        return f"❌ Titre introuvable : {title}"

    dists, idxs = knn.kneighbors(mat_df.loc[[isbn]], n_neighbors=k+1)
    recs = [
        [isbn2title[mat_df.index[i]], float(d)]
        for d, i in zip(dists[0][1:], idxs[0][1:])  # on saute l’indice 0 (le livre lui‑même)
    ]
    return [title, recs]
```

---

## 🧠 Pourquoi la distance cosinus ?

La **distance cosinus** mesure l'\*angle\* entre deux vecteurs plutôt que leur longueur :

```
          a · b
1 − -----------------
     ||a|| · ||b||
```

| Ce qu’elle mesure                           | Impact pour la recommandation                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Orientation des vecteurs (patron de votes)  | Ignore les notes trop hautes ou trop basses en moyenne ; ne retient que le profil relatif. |
| 0 ⇒ alignés • 1 ⇒ orthogonaux • 2 ⇒ opposés | Plus la valeur est proche de 0, plus deux livres partagent les mêmes lecteurs.             |

### Petit exemple

| Livre | U1 | U2 |
| ----- | -- | -- |
| A     | 8  | 4  |
| B     | 0  | 0  |
| C     | 2  | 1  |

Les vecteurs (8,0,2) et (4,0,1) sont colinéaires ⇒ distance 0, bien que U2 soit deux fois « moins généreux ».

### Cosinus vs Euclidienne

| Critère                         | Cosinus | Euclidienne                    |
| ------------------------------- | ------- | ------------------------------ |
| Insensible à l’échelle          | ✅       | ❌                              |
| Fonctionne sur matrices creuses | ✅       | ✅                              |
| Borne claire (0–2)              | ✅       | dépend du nombre de dimensions |

Ainsi, pour un filtrage collaboratif, la distance cosinus capture au mieux la **similarité de goûts** indépendamment de la tendance à surnoter ou sous‑noter.

---

## 📊 Vérifications visuelles optionnelles

```python
vc = ratings['user'].value_counts()
vc.plot.hist(bins=60, logy=True, title='Nombre de notes par utilisateur (log)')
```

Ajoute un histogramme équivalent pour les livres pour motiver les seuils 100/200.

---

## ⚠️ Limitations connues

| Problème                  | Commentaire                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| **Cold‑start**            | Un nouveau livre (< 100 notes) reste invisible.                       |
| **Recherche titre exact** | Besoin d’une correspondance 100 % (fuzzy‑matching possible).          |
| **Scalabilité**           | Le mode `brute` suffit < 50 k livres ; au‑delà, préférer FAISS/Annoy. |

---

## 🌱 Extensions possibles

* Remplacer K‑NN par une factorisation de matrice (SVD / ALS).
* Pondérer différemment le feedback implicite (notes 0).
* Pré‑calculer un index ANN pour les requêtes temps réel.
* Exposer via une API REST (FastAPI) ou une démo Streamlit.

---

## 📑 Références

1. Sarwar B. *et al.* « Item‑based Collaborative Filtering Recommendation Algorithms », 2001.
2. Certification *Machine Learning* de freeCodeCamp – Book Recommendation Engine.
3. Documentation `NearestNeighbors` de *scikit‑learn*.
