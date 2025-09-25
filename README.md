# Εργασία Ανάλυσης Φυσικής Γλώσσας 2025

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Frameworks](https://img.shields.io/badge/Frameworks-Transformers%20%7C%20spaCy%20%7C%20scikit--learn-orange)

**Ονοματεπώνυμο:** ΝΙΚΟΛΑΟΣ ΦΟΥΦΟΠΟΥΛΟΣ  
**ΑΜ:** Π22188

---

## 1. Επισκόπηση

Αυτό το repository περιέχει την υλοποίηση για την εργασία του μαθήματος "Ανάλυση Φυσικής Γλώσσας 2025". Ο σκοπός της εργασίας είναι η σημασιολογική ανακατασκευή αγγλικών κειμένων που περιέχουν γραμματικά και συντακτικά λάθη.

Για την επίλυση του προβλήματος, αναπτύχθηκαν και συγκρίθηκαν οι παρακάτω προσεγγίσεις:
-   Μια **custom, rule-based μέθοδος** σε Python.
-   Τρία **αυτοματοποιημένα pipelines** με χρήση των:
    1.  **LanguageTool** (βασισμένο σε κανόνες).
    2.  **PEGASUS** (μοντέλο παράφρασης).
    3.  **T5** (fine-tuned για διόρθωση γραμματικής).

Η αξιολόγηση της απόδοσης έγινε ποσοτικά, μέσω της **συνάφειας συνημιτόνου (cosine similarity)** με χρήση ενσωματώσεων λέξεων **GloVe** και οπτικά με τις τεχνικές **PCA** και **t-SNE**. Το bonus μέρος της εργασίας εξετάζει το πρόβλημα **Masked Clause Input** στην ελληνική γλώσσα, συγκρίνοντας τα μοντέλα **GreekBERT** και **XLM-RoBERTa**.

## 2. Δομή Αρχείων

Το repository είναι οργανωμένο σύμφωνα με τα παραδοτέα της εργασίας:

```
ErgasiaNPL/
├── Παραδοτέο_1/              # Scripts για την ανακατασκευή κειμένου
│   ├── A/                     # Υλοποίηση της custom, rule-based μεθόδου
│   │   └── custom_reconstruction.py
│   └── B/                     # Υλοποιήσεις των αυτόματων pipelines
│       ├── language_tool.py
│       ├── PEGASUS.py
│       └── t5.py
│
├── Παραδοτέο_2/              # Scripts για την υπολογιστική ανάλυση
│   └── cosine_analysis.py
│
├── Παραδοτέο_3/              # Script για την ανάλυση του bonus ερωτήματος
│   └── masked_clause_analysis.py
│   └── τεχνική_αναφορά.md      # Η τεχνική αναφορά της εργασίας
│
├── README.md                  # Το παρόν αρχείο
├── pyproject.toml             # Ορισμός του project και των dependencies
├── poetry.lock                # Αρχείο για αναπαραγώγιμες εγκαταστάσεις
└── .gitignore                 # Αρχείο για τον αποκλεισμό περιττών αρχείων
```

## 3. Εγκατάσταση & Setup

Για την εκτέλεση του project απαιτούνται τα παρακάτω:
-   [Git](https://git-scm.com/)
-   [Python 3.11](https://www.python.org/downloads/) ή νεότερη έκδοση
-   [Poetry](https://python-poetry.org/docs/#installation) για τη διαχείριση των dependencies

Ακολουθήστε τα παρακάτω βήματα για την πλήρη εγκατάσταση:

**1. Clone Git Repository**
```bash
  git clone https://github.com/nickfouf/ErgasiaNPL.git
```

**2. Μετάβαση στον Φάκελο του Project**
```bash
  cd ErgasiaNPL
```

**3. Εγκατάσταση Dependencies**  
Το Poetry θα δημιουργήσει αυτόματα ένα virtual environment και θα εγκαταστήσει όλες τις απαραίτητες βιβλιοθήκες που ορίζονται στο `pyproject.toml`.
```bash
  poetry install
```

## 4. Εκτέλεση

Όλα τα scripts πρέπει να εκτελούνται μέσω του Poetry για να χρησιμοποιηθεί το σωστό virtual environment.

**Σημείωση:** Κατά την πρώτη εκτέλεση ορισμένων scripts, ενδέχεται να γίνει αυτόματη λήψη των γλωσσικών μοντέλων (π.χ. T5, PEGASUS, GloVe).

### Παραδείγματα Εκτέλεσης

**Ανακατασκευή Κειμένου (π.χ. με το μοντέλο T5):**
```bash
  poetry run python Παραδοτέο_1/B/t5.py
```

**Υπολογιστική Ανάλυση (Cosine Similarity & Οπτικοποίηση):**
Το script αυτό θα υπολογίσει τις ομοιότητες και θα δημιουργήσει τα γραφήματα `text1_pca`, `text1_tsne`, `text2_pca` και `text2_tsne`.
```bash
  poetry run python Παραδοτέο_2/cosine_analysis.py
```

**Ανάλυση Bonus (Masked Clause Input):**
```bash
  poetry run python Παραδοτέο_3/masked_clause_analysis.py
  ```

## 5. Τεχνική Αναφορά

Η πλήρης τεχνική αναφορά, η οποία περιλαμβάνει τη μεθοδολογία, τα πειράματα, τα αποτελέσματα και τη συζήτηση, βρίσκεται στο αρχείο [τεχνική_αναφορά.md](Παραδοτέο_3/τεχνική_αναφορά.md) .