# Παραδοτέο 1
# Ερώτημα Α

import re

# Μικρά χρήσιμα λεξικά και λίστες για γρήγορη αναγνώριση λέξεων
# και διόρθωση των προτάσεων
small_verbs_dict = {
    "appreciate": ["appreciated", "appreciates", "appreciating"],
    "support": ["supported", "supports", "supporting"],
    "discuss": ["discussed", "discusses", "discussing"],
    "proceed": ["proceeded", "proceeds", "proceeding"],
    "publish": ["published", "publishes", "publishing"],
    "submit": ["submitted", "submits", "submitting"],
    "review": ["reviewed", "reviews", "reviewing"],
    "conclude": ["concluded", "concludes", "concluding"],
}

small_nouns_dict = {
    "support": ["support", "supports"],
    "publicate": ["publication", "publications"],
    "discuss": ["discussion", "discussions"],
    "proceed": ["proceeding", "proceedings"],
    "submit": ["submission", "submissions"],
    "review": ["review", "reviews"],
    "conclude": ["conclusion", "conclusions"],
}

small_adjectives_dict = {
    "appreciate": ["appreciative"],
    "support": ["supportive"],
}

small_articles_list = [
    "the", "a", "an", "this", "that", "these", "those",
]

small_preposition_list = [
    "of", "in", "on", "at", "for", "with", "about", "to", "from", "by", "as", "into",
    "through", "during", "before", "after", "between", "against", "without"
]


# Λίστα ρήματος "to be" για εύκολη αναγνώριση
be_verbs = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been']

# Βοηθητικός χάρτης για τη μετατροπή της ρίζας ενός ρήματος στην επιθυμητή ουσιαστική μορφή του.
base_verb_to_noun_form = {
    "discuss": "discussion",
    "submit": "submission",
    "conclude": "conclusion",
    "review": "review",
}
''
base_verb_to_preposition_map = {
    "appreciate": "of",
    "support": "of",
}

# Συνάρτηση να ελέγξουμε αν μια λέξη είναι ρήμα
def is_verb(word):
    return word in small_verbs_dict or any(word in forms for forms in small_verbs_dict.values())


# Συνάρτηση να ελέγξουμε αν μια λέξη είναι ουσιαστικό
def is_noun(word):
    return word in small_nouns_dict or any(word in forms for forms in small_nouns_dict.values())


# Συνάρτηση να ελέγξουμε αν μια λέξη είναι επίθετο
def is_adjective(word):
    return word in small_adjectives_dict or any(word in forms for forms in small_adjectives_dict.values())


# Συνάρτηση για εύρεση της ρίζας ενός ρήματος
def find_base_verb(word, verbs_dict):
    for base, forms in verbs_dict.items():
        if word == base or word in forms:
            return base
    return None


# Συνάρτηση για εύρεση της ρίζας ενός ουσιαστικού
def find_base_noun(word, nouns_dict):
    for base, forms in nouns_dict.items():
        if word == base or word in forms:
            return base
    return None


# Συνάρτηση για εύρεση της ρίζας ενός επιθέτου
def find_base_adjective(word, adjectives_dict):
    for base, forms in adjectives_dict.items():
        if word == base or word in forms:
            return base
    return None


# Απλή συνάρτηση για tokenization
def tokenize(text):
    pattern = r'[a-zA-Z]+| +|\.|[^a-zA-Z. ]+'
    tokens = re.findall(pattern, text)
    result = []
    for t in tokens:
        lower_t = t.lower()
        if t.isspace():
            result.append({"type": "space", "value": t})
        elif t == '.':
            result.append({"type": "punctuation", "value": t})
        elif re.fullmatch(r'[a-zA-Z]+', t):
            if is_verb(lower_t):
                result.append({"type": "verb", "value": t, "base": find_base_verb(lower_t, small_verbs_dict)})
            elif is_noun(lower_t):
                result.append({"type": "noun", "value": t, "base": find_base_noun(lower_t, small_nouns_dict)})
            elif is_adjective(lower_t):
                result.append(
                    {"type": "adjective", "value": t, "base": find_base_adjective(lower_t, small_adjectives_dict)})
            elif lower_t in small_articles_list:
                result.append({"type": "article", "value": t})
            elif lower_t in small_preposition_list:
                result.append({"type": "preposition", "value": t})
            else:
                result.append({"type": "word", "value": t})
        else:
            result.append({"type": "other", "value": t})
    return result


# Συνάρτηση για ανακατασκευή της πρότασης από τα tokens
def reconstruct(tokens):
    reconstructed = []
    for token in tokens:
        reconstructed.append(token['value'])
    return ''.join(reconstructed).strip()

# Συνάρτηση για εύρεση του επόμενου σημαντικού token
# (Δηλαδή, αγνοεί τα κενά και άλλα μη σημαντικά tokens)
def find_next_significant_token(tokens, start_index):
    for i in range(start_index, len(tokens)):
        token_type = tokens[i].get("type")
        if token_type not in ["space", "other"]:
            return tokens[i], i
    return None, -1

# Συνάρτηση για την εύρεση ενός συγκεκριμένου token (εφόσον υπάρχει παρακάτω)
def find_next_token(tokens, start_index, token_type):
    for i in range(start_index, len(tokens)):
        if tokens[i].get("type") == token_type:
            return tokens[i], i
    return None, -1


def manual_correct(tokens):
    corrected = []
    i = 0

    while i < len(tokens):
        current_token = tokens[i]

        # === ΚΑΝΟΝΑΣ 1 (Γενικευμένος): Διόρθωση "[be_verb] ... [verb]" -> "[be_verb] ... [adjective] [preposition]" ===
        # Παράδειγμα: "am very appreciated" -> "am very appreciative of"
        if current_token['value'].lower() in be_verbs:
            next_token, next_index = find_next_significant_token(tokens, i + 1)
            # Βρίσκουμε το επόμενο token που είναι ρήμα το οποίο μπορούμε να μετατρέψουμε
            while next_token and next_token['type'] != 'verb' or (next_token.get('base') not in small_adjectives_dict):
                next_token, next_index = find_next_significant_token(tokens, next_index + 1)

            if next_token:  # Αν βρήκαμε ένα μετατρέψιμο ρήμα
                # Προσθέτουμε τα πάντα ανάμεσα στο ρήμα 'be' και το λανθασμένο ρήμα
                corrected.extend(tokens[i:next_index])

                # Αντικαθιστούμε το ρήμα με την επίθετη μορφή του
                base_verb = next_token.get('base')
                adjective_form = small_adjectives_dict[base_verb][0]
                corrected.append({'type': 'adjective', 'value': adjective_form, 'base': base_verb})

                # Ελέγχουμε αν πρέπει να προστεθεί η απαιτούμενη πρόθεση
                required_preposition = base_verb_to_preposition_map.get(base_verb)
                if required_preposition:
                    token_after, _ = find_next_significant_token(tokens, next_index + 1)
                    if not token_after or token_after['value'].lower() != required_preposition:
                        corrected.append({'type': 'space', 'value': ' '})
                        corrected.append({'type': 'preposition', 'value': required_preposition})

                i = next_index + 1
                continue

        # === ΚΑΝΟΝΑΣ 2 (Γενικευμένος): Διόρθωση "[preposition] ... [verb]" -> "[preposition] ... [noun]" ===
        # Παράδειγμα: "during our final discuss" -> "during our final discussion"
        if current_token['type'] == 'preposition':
            next_token, next_index = find_next_token(tokens, i + 1, 'verb')
            if next_token and next_token['type'] == 'verb':
                base_verb = find_base_verb(next_token['base'], small_verbs_dict)
                if base_verb in base_verb_to_noun_form:
                    noun_form = base_verb_to_noun_form[base_verb]
                    for j in range(i, next_index):
                        corrected.append({'type': tokens[j]['type'], 'value': tokens[j]['value']})
                    corrected.append({'type': 'noun', 'value': noun_form, 'base': base_verb})
                    i = next_index + 1
                    continue

        # === Default case (Απλή προσθήκη του token) ===
        corrected.append(current_token)
        i += 1

    # Ελέγχουμε αν έχει μπει τελικό σημείο στίξης
    if corrected and corrected[-1]['type'] != 'punctuation':
        corrected.append({'type': 'punctuation', 'value': '.'})

    return corrected


sentence1 = "I am very appreciated the full support of the professor, for our Springer proceedings publication"
sentence2 = "During our final discuss, I told him about the new submission."

# Διαχωρισμός σε tokens
tokens1 = tokenize(sentence1)
tokens2 = tokenize(sentence2)

# Εφαρμογή διορθώσεων
corrected_tokens1 = manual_correct(tokens1)
corrected_tokens2 = manual_correct(tokens2)

# Ανακατασκευή
print("Πρόταση 1:")
print("Πρωτότυπο:", sentence1)
print("Διορθωμένο:", reconstruct(corrected_tokens1))
print("\nΠρόταση 2:")
print("Πρωτότυπο:", sentence2)
print("Διορθωμένο:", reconstruct(corrected_tokens2))