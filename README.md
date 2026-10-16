# DocChat : questions-réponses sur tes documents (RAG)

DocChat est un assistant qui répond à des questions **à partir du contenu de tes propres PDF** (cours, rapports,
textes réglementaires...). Tu charges un ou plusieurs documents, tu coches ceux à interroger, tu poses ta question :
la réponse est rédigée par un modèle de langage à partir des passages retrouvés, avec les pages sources et un
contrôle de fiabilité.

![Capture d'écran](docs/DocChat.png) 

## Sommaire

1. [Fonctionnalités](#fonctionnalités)
2. [Comment ça marche](#comment-ça-marche)
3. [Structure du projet](#structure-du-projet)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
6. [Configuration](#configuration)
7. [Évaluation](#évaluation)
8. [Choix techniques et leçons apprises](#choix-techniques-et-leçons-apprises)
9. [Limites connues](#limites-connues)
10. [Confidentialité et sécurité](#confidentialité-et-sécurité)
11. [Dépannage](#dépannage)
12. [Pistes d'amélioration](#pistes-damélioration)

---

## Fonctionnalités

- **Chargement de PDF depuis l'interface** : un ou plusieurs documents, jusqu'à 5 par session et 300 pages chacun.
- **Choix des documents actifs** : une case à cocher par document. Un seul coché pour des réponses ciblées,
  plusieurs pour croiser les sources.
- **Indexation unique** : chaque document est analysé une seule fois, puis réutilisé depuis un cache sur disque.
- **Recherche hybride** : par le sens (FAISS) et par les mots exacts (BM25), fusionnées par RRF.
- **Réponses sourcées** : pour chaque réponse, les documents et pages utilisés, et les extraits exacts.
- **Contrôle de fiabilité** : le modèle doit citer une phrase des extraits, et on vérifie qu'elle y figure vraiment.
  Un badge indique si la réponse est appuyée par le document (✅), à vérifier (⚠️), ou sans réponse trouvée (ℹ️).
- **Deux modèles de langage au choix** : Ollama (local, gratuit) ou Groq (API avec quotas gratuits).
- **Script d'évaluation** : mesure la qualité de la recherche sur tes propres questions et aide à régler le seuil de refus.

---

## Comment ça marche

Le système fonctionne en deux temps : l'**indexation** (une fois par document) et la **réponse** (à chaque question).

### Phase 1 : indexation d'un document

```
PDF
 |
 v
1. Lecture du texte, page par page              (pdfplumber)
 |
 v
2. Nettoyage : suppression des en-têtes, pieds de page et numéros de page,
   rejet des tables des matières, rejet des PDF scannés
 |
 v
3. Découpage en passages ("chunks") par sections
 |
 v
4. Calcul d'un vecteur par passage              (sentence-transformers)
 |
 v
5. Sauvegarde dans index/docs/<identifiant>/    (passages + vecteurs)
```

### Phase 2 : réponse à une question

```
Question
 |
 +--> vecteur de la question --> FAISS (similarité de sens) ---+
 |                                                             +--> fusion RRF --> meilleurs passages
 +--> mots de la question -----> BM25  (mots exacts) ----------+            |
                                                                             v
                                         + passages voisins (avant / après le meilleur)
                                                                             |
                                                                             v
                          contexte = extraits regroupés + question --> modèle de langage
                                                                             |
                                                                             v
                                              CITATION + RÉPONSE --> contrôle de la citation
                                                                             |
                                                                             v
                                         réponse + badge (✅ / ⚠️ / ℹ️) + pages sources
```

### Détail des étapes

**1. Lecture et nettoyage (`src/ingest.py`)**
- Le texte est extrait avec `pdfplumber` avec un seuil d'espacement `x_tolerance = 2`. Avec la valeur par défaut (3),
  certains PDF produits par LaTeX donnent des mots collés (`Sansmécanisme`), ce qui ruine la recherche par mots.
- Les lignes qui se répètent sur au moins 40 % des pages (pied de page d'un diaporama, numéro de page) sont supprimées.
  Les chiffres sont neutralisés pour que « 7/59 » et « 12/59 » comptent comme la même ligne.
- Les tables des matières (suites de « . . . . . ») sont ignorées.
- Un PDF qui ne contient presque aucun texte est refusé avec un message clair : c'est probablement un scan.

**2. Découpage par sections (`src/ingest.py`)**
- Un nouveau passage commence à chaque **titre** détecté (« Chapitre 3 », « 2.1.3 Les types d'exceptions »,
  « 1 Les curseurs »...). Les sections sont regroupées tant que le passage reste sous `CHUNK_SIZE` (1000 caractères).
- Une section plus longue que cette taille est coupée entre les phrases, avec un chevauchement de 150 caractères.
- Résultat : une liste ou une classification courte reste d'un seul tenant au lieu d'être coupée en deux.
- Chaque passage garde son document, sa page et sa position (`idx`) dans le document.

**3. Vecteurs (`src/embedder.py`)**
- Modèle : `paraphrase-multilingual-MiniLM-L12-v2` (multilingue, vecteurs de 384 nombres).
- Les vecteurs sont normalisés : le produit scalaire équivaut alors à la similarité cosinus.

**4. Recherche hybride (`src/retrieve.py`)**
- **FAISS** (`IndexFlatIP`) retrouve les passages au sens proche de la question.
- **BM25** retrouve les passages qui contiennent les mêmes mots, ce qui aide pour les termes techniques
  (`DEALLOCATE`, `DUP_VAL_ON_INDEX`). La tokenisation garde les underscores et retire les mots vides.
- La fusion **RRF** (Reciprocal Rank Fusion) donne à chaque passage `poids / (60 + rang)` dans chaque classement,
  puis additionne. Un passage bien placé dans les deux classements monte en tête.
- Les 4 meilleurs passages sont retenus (`TOP_K`). Pour le meilleur d'entre eux (`NEIGHBOR_TOP`), on ajoute aussi
  le passage qui précède et celui qui suit dans le même document.

**5. Génération (`src/generate.py`, `src/llm.py`)**
- Les passages consécutifs sont recollés en blocs (le chevauchement est retiré), le plus pertinent en premier.
- Le modèle reçoit une consigne stricte : n'utiliser que les extraits, ne pas inventer de catégories, présenter
  séparément des classifications différentes, et répondre au format `CITATION:` puis `REPONSE:`.
- Température 0 pour des réponses stables.

**6. Contrôle de la réponse (`src/generate.py`)**
- La citation est cherchée dans les extraits (plus longue sous-chaîne commune, seuil de 75 %).
- On mesure aussi la part des mots importants de la réponse qui figurent dans les extraits (seuil de 50 %).
- Statuts : **ok** (citation retrouvée et mots présents), **warn** (à vérifier), **none** (rien trouvé).
- Le **mode strict** remplace une réponse « à vérifier » par un refus.
- Le seuil `MIN_SIM` permet de refuser sans même appeler le modèle quand aucun passage n'est proche de la question.

**7. Cache et sessions (`src/store.py`)**
- Chaque document est identifié par l'empreinte de son contenu (SHA-1) : le même fichier n'est indexé qu'une fois,
  même s'il est renommé.
- Le cache enregistre la configuration de découpage : si tu modifies `CHUNK_SIZE`, le modèle d'embeddings ou la
  version du découpage, l'index est refait automatiquement.
- La liste des documents et les cases cochées sont propres à chaque session : un autre utilisateur ne voit pas
  tes documents.

---

## Structure du projet

```
rag_chatbot/
├── app.py                  Interface Streamlit (chargement, choix des documents, discussion)
├── requirements.txt        Dépendances Python
├── README.md               Ce fichier
├── .gitignore              Exclut venv, index, PDF et clés
│
├── .streamlit/
│   └── config.toml         Thème de l'interface (sombre, accent bleu)
│
├── src/
│   ├── config.py           Tous les réglages au même endroit
│   ├── ingest.py           Lecture des PDF, nettoyage, découpage en passages
│   ├── embedder.py         Texte -> vecteurs (+ faux embedder pour les tests)
│   ├── store.py            Documents, index par document, cache sur disque, moteurs de recherche
│   ├── retrieve.py         Recherche hybride FAISS + BM25 + RRF, passages voisins
│   ├── llm.py              Appel au modèle de langage : Ollama ou Groq
│   └── generate.py         Contexte, consigne, contrôle de la citation
│
├── eval/
│   ├── questions.json      Questions de test et mots-clés attendus
│   └── evaluate.py         Mesure de la recherche et aide au réglage de MIN_SIM
│
├── data/
│   └── raw/                PDF de démonstration et d'évaluation (non versionnés)
│
└── index/                  Créé automatiquement : un dossier par document indexé (non versionné)
    └── docs/<id>/
        ├── meta.json       Nom, nombre de pages, signature de la configuration
        ├── chunks.json     Passages (texte, page, position)
        └── embeddings.npy  Vecteurs des passages
```

**Rôle de chaque module**

| Fichier | Responsabilité | Dépend de |
|---|---|---|
| `config.py` | Réglages (tailles, poids, modèles, limites) | rien |
| `ingest.py` | PDF -> pages -> passages nettoyés | `config` |
| `embedder.py` | Passages et questions -> vecteurs | `config` |
| `retrieve.py` | `Searcher` : classement hybride et voisins | `config` |
| `store.py` | `Store` : ajoute un PDF, gère le cache, fournit un `Searcher` | `ingest`, `retrieve` |
| `llm.py` | `chat()` : un seul point d'entrée pour Ollama ou Groq | `config` |
| `generate.py` | `answer()` : recherche, consigne, appel du modèle, contrôle | `retrieve`, `llm` |
| `app.py` | Interface | tous les modules |
| `eval/evaluate.py` | Évaluation hors interface | `store`, `embedder` |

---

## Installation

### Prérequis

- Python récent (3.10 ou plus conseillé)
- Environ 8 Go de RAM pour le mode local (Ollama avec un modèle de 3 milliards de paramètres)
- Des PDF dont le texte est sélectionnable (pas des scans)

### Étapes

```bash
# 1. Créer et activer un environnement virtuel
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux / Mac

# 2. Installer les dépendances
pip install -r requirements.txt
```

Au premier calcul de vecteurs, le modèle d'embeddings est téléchargé automatiquement (quelques centaines de Mo).
Un avertissement sur les « requêtes non authentifiées » du Hugging Face Hub peut apparaître : il est sans gravité.

### Mode local : Ollama

1. Installe Ollama depuis [ollama.com](https://ollama.com). À la question sur la création d'un compte,
   le mode local n'en a pas besoin.
2. Télécharge un modèle :

```bash
ollama pull qwen2.5:3b
```

### Mode en ligne : Groq

1. Crée un compte sur [console.groq.com](https://console.groq.com) et génère une clé API.
2. Colle la clé dans la barre latérale de l'application, ou définis la variable d'environnement `GROQ_API_KEY`.

Les quotas gratuits sont limités et peuvent changer : consulte ta console. La liste des modèles proposés dans
l'application est récupérée en direct auprès de Groq.

---

## Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

Lance toujours la commande depuis la racine du projet (les chemins sont relatifs).

### Dans l'interface

1. **Barre latérale** : choisis le modèle de langage (Ollama sur ce PC, ou Groq). Les « Réglages avancés » donnent
   le nombre de passages retrouvés et le mode strict.
2. **Ajouter des documents** : dépose un ou plusieurs PDF. Une barre de progression s'affiche pendant l'indexation.
   Les PDF placés dans `data/raw/` apparaissent dans « Documents de démonstration ».
3. **Documents à interroger** : coche ceux qui doivent servir à répondre. Le bouton ✖ retire un document de la session.
4. **Discussion** : pose ta question. Sous chaque réponse, tu vois le badge de fiabilité, les pages sources,
   le temps de réponse, et un dépliant avec les extraits exacts utilisés.

### Lire les badges

| Badge | Signification |
|---|---|
| ✅ Réponse appuyée par le document | La citation du modèle a été retrouvée dans les extraits et ses mots y figurent |
| ⚠️ À vérifier | La citation est introuvable ou la réponse contient beaucoup de mots absents des extraits |
| ℹ️ Rien de pertinent trouvé | Aucune information utile dans les documents actifs |

Un ⚠️ ne veut pas dire que la réponse est fausse, mais qu'elle n'est pas démontrée par les extraits.
Compare-la avec le dépliant « Voir les extraits utilisés ».

---

## Configuration

Tous les réglages sont dans `src/config.py`.

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `RAW_DIR` | `data/raw` | Dossier des PDF de démonstration et d'évaluation |
| `CACHE_DIR` | `index/docs` | Dossier du cache d'indexation |
| `MAX_PAGES` | 300 | Pages maximum par document chargé |
| `MAX_DOCS` | 5 | Documents maximum par session |
| `CHUNK_SIZE` | 1000 | Taille maximale d'un passage (caractères) |
| `CHUNK_OVERLAP` | 150 | Chevauchement quand une section est coupée |
| `MIN_CHUNK` | 60 | Les passages plus courts sont ignorés |
| `X_TOLERANCE` | 2 | Seuil d'espacement de pdfplumber |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embeddings |
| `TOP_K` | 4 | Passages retrouvés par question |
| `NEIGHBOR_TOP` | 1 | Nombre de meilleurs passages dont on ajoute les voisins |
| `BM25_WEIGHT` | 1.0 | Poids de la recherche par mots exacts (0 = désactivée) |
| `MIN_SIM` | 0.0 | Seuil de similarité sous lequel on refuse sans appeler le modèle (0 = désactivé) |
| `LLM_BACKEND` | `ollama` | Modèle de langage par défaut : `ollama` ou `groq` |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modèle Ollama |
| `OLLAMA_NUM_CTX` | 4096 | Fenêtre de contexte demandée à Ollama |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Modèle Groq par défaut (à vérifier : la liste évolue) |
| `MAX_ANSWER_TOKENS` | 700 | Longueur maximale de la réponse générée |
| `STRICT_GROUNDING` | `False` | Remplacer par un refus les réponses non appuyées par les extraits |

**Variables d'environnement**

| Variable | Usage |
|---|---|
| `GROQ_API_KEY` | Clé Groq, pré-remplit le champ de la barre latérale |
| `LLM_BACKEND` | Choisit le modèle de langage par défaut |
| `RAG_EMBEDDER=hash` | Mode test : remplace le modèle d'embeddings par un faux (aucun téléchargement) |

**Conseils de réglage**
- Réponses trop lentes avec Ollama : baisse « Passages retrouvés » à 3 dans l'interface.
- Réponses coupées en plein mot : augmente `MAX_ANSWER_TOKENS`.
- Termes techniques mal retrouvés : garde `BM25_WEIGHT` élevé.
- Après avoir modifié le découpage, relance simplement l'application : l'index se refait tout seul.

---

## Évaluation

### Principe

`eval/evaluate.py` mesure la **qualité de la recherche**, pas celle du texte généré. Pour chaque question de
`eval/questions.json`, il regarde si les mots-clés de la bonne réponse se trouvent dans les passages retrouvés.
> **Note sur les données** : les PDF du corpus de test ne sont pas fournis (ils appartiennent à leurs auteurs).
> `eval/questions.json` sert d'exemple du format. Pour reproduire l'évaluation, place tes propres PDF dans
> `data/raw/` et écris des questions adaptées.

Format de `eval/questions.json` :

```json
[
  {"question": "C'est quoi un curseur ?", "contient": ["objet de base de données"]},
  {"question": "Quelles sont les étapes d'exécution d'une application YARN ?",
   "contient": ["soumet une application", "libère les ressources et se termine"]},
  {"question": "C'est qui le président du Togo ?", "contient": []}
]
```

- `contient` : des mots ou expressions copiés du PDF, en minuscules ou non (la comparaison ignore la casse
  et les espaces multiples). Tous doivent figurer dans les passages retrouvés.
- `contient` vide : question **hors sujet**, ignorée pour les scores, mais utilisée pour régler le seuil `MIN_SIM`.
- Choisis des expressions courtes et distinctives (1 à 5 mots).

### Lancer

Mets les PDF dans `data/raw/`, puis :

```bash
python eval/evaluate.py
```

La première fois, l'indexation prend du temps ; ensuite elle est instantanée grâce au cache.

### Ce que le script affiche

Deux situations, chacune avec trois poids de BM25 (0, 0.5, 1.0) :
- **Tous les documents actifs** : la situation réelle quand plusieurs documents sont cochés.
- **Le document concerné seul** : la recherche limitée au document qui contient la réponse.

Pour chaque ligne :
- `@1`, `@3`, `@5`, `@10` : part des questions dont les mots-clés se trouvent dans les 1, 3, 5 ou 10 premiers passages ;
- `contexte envoyé au modèle` : même mesure sur ce que le modèle reçoit réellement (meilleurs passages + voisins),
  avec la taille moyenne du contexte en caractères.

Il liste ensuite les questions dont le contexte ne contient pas la réponse, puis aide à régler `MIN_SIM`
en comparant la similarité des questions du cours à celle des questions hors sujet.

### Résultats

À compléter avec tes propres mesures :

| Situation | BM25 | @1 | @3 | @5 | @10 | Contexte envoyé |
|---|---|---|---|---|---|---|
| Tous les documents | 0 | | | | | |
| Tous les documents | 0.5 | | | | | |
| Tous les documents | 1.0 | | | | | |
| Document concerné seul | 1.0 | | | | | |

Corpus de test : 3 PDF (environ 200 pages, 275 passages) : un exposé sur les curseurs et les exceptions,
un cours d'administration des bases de données, un cours sur Hadoop.

### Précautions d'interprétation

- Le jeu de test est petit : une question de plus ou de moins déplace le score de quelques points.
- Si le poids de BM25 est choisi sur les mêmes questions que celles mesurées, le score est un peu optimiste.
- Le test vérifie la présence de mots-clés, pas la justesse de la réponse rédigée.

---

## Choix techniques et leçons apprises

- **L'extraction du texte est la base de tout.** Des mots collés et des pieds de page répétés dans les passages
  dégradaient autant la recherche par le sens que celle par les mots. Régler `x_tolerance` et nettoyer les
  lignes répétées a eu plus d'effet que n'importe quel réglage de recherche.
- **La recherche hybride a besoin d'un texte propre.** Testée sur un texte aux mots collés, BM25 faisait baisser
  les scores ; sur le texte nettoyé, il les fait monter. Le test sur les deux situations a permis de le voir.
- **Un découpage à taille fixe coupe les listes.** Une section « types d'exceptions » coupée en deux donnait
  une réponse qui mélangeait deux classifications. Le découpage par sections règle ce cas.
- **Mesurer avant de régler.** Chaque changement (poids de BM25, découpage, nettoyage) a été comparé sur le même jeu
  de questions. Une première version du test comptait une « bonne page » au lieu d'un « bon passage », ce qui
  rendait les scores trop optimistes : il a fallu le corriger.
- **Un petit modèle local obéit moins bien.** Un modèle de 3 milliards de paramètres reformule, mélange ou ajoute
  des informations. D'où le contrôle de citation, et l'option Groq pour un modèle plus grand.
- **Fenêtre de contexte d'Ollama.** Si elle est trop petite, le début de la consigne peut être coupé sans erreur ;
  `OLLAMA_NUM_CTX` la fixe explicitement.

---

## Limites connues

- Les PDF scannés (images) ne sont pas lus : il faudrait une étape d'OCR.
- Seul le texte est analysé : les tableaux, schémas et images ne sont pas compris.
- Les questions qui demandent de rassembler de nombreux passages éloignés (synthèse d'un document entier)
  sont mal servies : le système retrouve quelques passages, pas le document complet.
- Le contrôle de citation réduit les inventions sans les supprimer : un modèle peut citer une vraie phrase
  puis répondre autre chose. Le badge ✅ n'est pas une garantie.
- Avec Ollama sur processeur, une réponse prend plusieurs secondes à plusieurs dizaines de secondes.
- Pas de mémoire de conversation : chaque question est traitée indépendamment des précédentes.
- Une seule personne à la fois en mode Ollama local, puisque le modèle tourne sur la machine qui héberge l'application.

---

## Confidentialité et sécurité

- **Avec Ollama**, tout reste sur ta machine.
- **Avec Groq**, le texte des passages et ta question sont envoyés à un service tiers. N'utilise pas ce mode pour
  des documents confidentiels.
- **Clé API** : ne l'écris jamais dans un fichier du projet. Utilise le champ de l'interface ou la variable
  d'environnement `GROQ_API_KEY`. Le `.gitignore` exclut `.env`.
- **Cache** : les documents chargés sont conservés sur le disque du serveur dans `index/` (texte des passages et
  vecteurs). Supprime ce dossier pour tout effacer. Il est exclu de Git.
- **PDF personnels** : `data/raw/*.pdf` est exclu de Git pour ne pas publier de documents dont tu n'as pas les droits.

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| « Ollama ne répond pas » | Ollama n'est pas lancé, ou le modèle n'est pas téléchargé | Lance l'application Ollama, puis `ollama pull qwen2.5:3b` |
| Réponse coupée en plein mot | Limite de longueur atteinte | Augmente `MAX_ANSWER_TOKENS` dans `config.py` |
| « Ce PDF est probablement un scan » | Pas de texte sélectionnable | Utilise un PDF avec du texte, ou fais passer le document par un OCR |
| Erreur 429 avec Groq | Quota de requêtes atteint | Attends un peu, puis réessaie |
| Erreur 401 avec Groq | Clé refusée | Vérifie la clé dans la console Groq |
| Première question très lente | Chargement du modèle d'embeddings et du modèle Ollama en mémoire | Normal, les suivantes sont plus rapides |
| Document refusé : trop long | Plus de 300 pages | Découpe le PDF ou augmente `MAX_PAGES` |
| Résultats qui changent après une modification | L'index est refait quand la configuration de découpage change | Normal, attends la fin de l'indexation |
| Erreur d'import au lancement | Commande lancée hors de la racine ou dépendances manquantes | Lance depuis la racine et refais `pip install -r requirements.txt` |

---

## Pistes d'amélioration

- **Tester les réponses générées** (et pas seulement la recherche) : mots-clés présents, refus des questions hors
  sujet, part de ✅ et de ⚠️, comparaison Ollama et Groq.
- **Reranker** : réordonner les passages retrouvés avec un modèle dédié.
- **Modèle d'embeddings plus performant** pour le français.
- **Mémoire de conversation** pour les questions de suivi.
- **Affichage de la réponse en continu** (streaming) pour réduire l'attente avec Ollama.
- **OCR** pour les PDF scannés, et prise en charge d'autres formats (DOCX, TXT).
- **Déploiement en ligne** : exige un hébergement avec assez de mémoire pour le modèle d'embeddings, et un modèle
  de langage accessible par API.

---

## Auteur

< YENUI Luther Maxwell> < Etudiant en Intelligence artificielle et Big data >

