import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import re
from tabulate import tabulate
import requests
from tqdm import tqdm
import zipfile

# Πρωτότυπο κείμενο 1
original_text_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

# Χειροκίνητη ανακατασκευή κειμένου 1
reconstructed_text_1_custom = "I am very appreciative of the full support of the professor, for our Springer proceedings publication."

# Ανακατασκευή κειμένου 1 από Language Tool
reconstructed_text_1_lt = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

# Ανακατασκευή κειμένου 1 από PEGASUS
reconstructed_text_1_pegasus = """Our Chinese culture has a dragon boat festival that is celebrated today. Hope you enjoy it as much as I did.
Thank you for sending a message to the doctor so that he could check us out.
I received the message to see the approved one. I received the message from the professor a couple of days ago.
I would like to thank the professor for his full support for our Springer proceedings publication."""

# Ανακατασκευή κειμένου 1 από T5
reconstructed_text_1_t5 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safety and great in our lives.
Hope you too, to enjoy it as my deepest wishes.
Thank you for your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago.
I am very appreciated the full support of the professor, for our Springer proceedings publication."""


# Πρωτότυπο κείμενο 2
original_text_2 = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""

# Χειροκίνητη ανακατασκευή κειμένου 2
reconstructed_text_2_custom = "During our final discussion, I told him about the new submission."

# Ανακατασκευή κειμένου 2 από Language Tool
reconstructed_text_2_lt = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates were confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although a bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plans for the acknowledgments section edit before
he's sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""

# Ανακατασκευή κειμένου 2 από PEGASUS
reconstructed_text_2_pegasus = """I told him about the new submission we were waiting for, but the updates were confusing as they didn't include the full feedback from the reviewer or editor.
I think the team tried their best for paper and cooperation despite the recent delay and less communication.
We should be thankful for the acceptance and efforts until the Springer link came last week, I think.
If the doctor still plans for the acknowledgments section to be edited before he sends again, please remind me.
I apologize if I missed that part final. Let's make sure all are safe and celebrate the outcome with coffee and targets."""

# Ανακατασκευή κειμένου 2 από T5
reconstructed_text_2_t5 = """During our final discussion, I told him about the new submission — the one we were waiting for since last autumn, but the updates were confusing as it not included the full feedback from reviewer or maybe editor?
Anyway, I believe the team, although a bit delayed and less communication at recent days, they really tried best for paper and cooperation.
We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plans for the acknowledgments section edit before he sends again.
Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."""

texts_to_compare_1 = {
    "Πρωτότυπο": original_text_1,
    "Custom (Απόσπασμα)": reconstructed_text_1_custom,
    "LanguageTool": reconstructed_text_1_lt,
    "PEGASUS": reconstructed_text_1_pegasus,
    "T5": reconstructed_text_1_t5,
}

texts_to_compare_2 = {
    "Πρωτότυπο": original_text_2,
    "Custom (Απόσπασμα)": reconstructed_text_2_custom,
    "LanguageTool": reconstructed_text_2_lt,
    "PEGASUS": reconstructed_text_2_pegasus,
    "T5": reconstructed_text_2_t5,
}


def load_glove_model_from_web(glove_zip_url="http://nlp.stanford.edu/data/glove.6B.zip", embedding_dim=100):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "glove_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    file_name = f"glove.6B.{embedding_dim}d.txt"
    unzipped_path = os.path.join(cache_dir, file_name)
    zip_path = os.path.join(cache_dir, "glove.6B.zip")

    if not os.path.exists(unzipped_path):
        if not os.path.exists(zip_path):
            print(f"Γίνεται λήψη του GloVe μοντέλου ({glove_zip_url})...")
            response = requests.get(glove_zip_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, "wb") as f, tqdm(
                    desc="Downloading", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            print("Η λήψη ολοκληρώθηκε.")
        print(f"Αποσυμπίεση του αρχείου {file_name} στο {cache_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(file_name, path=cache_dir)
        print("Η αποσυμπίεση ολοκληρώθηκε.")

    print(f"Φόρτωση του GloVe μοντέλου από το {unzipped_path}...")
    model = {}
    with open(unzipped_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    print(f"Το μοντέλο GloVe φορτώθηκε με επιτυχία. Βρέθηκαν {len(model)} λέξεις.")
    return model


def get_text_vector(text, model):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    vector_size = len(next(iter(model.values())))
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def visualize_embeddings(texts_dict, model, method='pca', title_prefix=''):
    plt.figure(figsize=(14, 10))
    all_words = []
    for text in texts_dict.values():
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        for word in words:
            if word in model:
                all_words.append(word)
    unique_words = sorted(list(set(all_words)))
    if not unique_words:
        print("Δεν βρέθηκαν λέξεις για οπτικοποίηση.")
        return
    word_vectors = np.array([model[word] for word in unique_words])

    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = f'{title_prefix} Οπτικοποίηση Ενσωματώσεων Λέξεων (PCA)'
    elif method.lower() == 'tsne':
        perplexity_value = min(30, len(unique_words) - 1)
        if perplexity_value <= 0:
            print(f"Δεν υπάρχουν αρκετές λέξεις για t-SNE (βρέθηκαν {len(unique_words)}).")
            return
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='pca', learning_rate='auto')
        title = f'{title_prefix} Οπτικοποίηση Ενσωματώσεων Λέξεων (t-SNE)'
    else:
        raise ValueError("Μη έγκυρη μέθοδος: επιλέξτε 'pca' ή 'tsne'")

    reduced_vectors = reducer.fit_transform(word_vectors)

    cmap = plt.get_cmap('viridis', len(texts_dict.keys()))
    label_to_color = {label: cmap(i) for i, label in enumerate(texts_dict.keys())}
    for i, word in enumerate(unique_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], color='gray', alpha=0.6, s=100)
        plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02, word, fontsize=10)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for
               label, color in label_to_color.items()]

    plt.title(title)
    plt.xlabel(f'{method.upper()} Συνιστώσα 1')
    plt.ylabel(f'{method.upper()} Συνιστώσα 2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    glove_model = load_glove_model_from_web(embedding_dim=100)

    if glove_model:
        print("\n" + "=" * 80)
        print("ΑΝΑΛΥΣΗ ΓΙΑ ΤΟ ΚΕΙΜΕΝΟ 1")
        print("=" * 80)

        original_vector_1 = get_text_vector(original_text_1, glove_model).reshape(1, -1)
        results_data_1 = []

        for name, text in texts_to_compare_1.items():
            if name == "Πρωτότυπο":
                continue
            reconstructed_vector = get_text_vector(text, glove_model).reshape(1, -1)
            cos_sim = cosine_similarity(original_vector_1, reconstructed_vector)[0][0]
            results_data_1.append([f"Πρωτότυπο vs. {name}", cos_sim])

        print("\nΑποτελέσματα Cosine Similarity για το Κείμενο 1:")
        print(tabulate(results_data_1, headers=["Σύγκριση", "Similarity Score"], floatfmt=".4f", tablefmt="github"))

        print("\n" + "=" * 80)
        print("ΑΝΑΛΥΣΗ ΓΙΑ ΤΟ ΚΕΙΜΕΝΟ 2")
        print("=" * 80)

        original_vector_2 = get_text_vector(original_text_2, glove_model).reshape(1, -1)
        results_data_2 = []

        for name, text in texts_to_compare_2.items():
            if name == "Πρωτότυπο":
                continue
            reconstructed_vector = get_text_vector(text, glove_model).reshape(1, -1)
            cos_sim = cosine_similarity(original_vector_2, reconstructed_vector)[0][0]
            results_data_2.append([f"Πρωτότυπο vs. {name}", cos_sim])

        print("\nΑποτελέσματα Cosine Similarity για το Κείμενο 2:")
        print(tabulate(results_data_2, headers=["Σύγκριση", "Similarity Score"], floatfmt=".4f", tablefmt="github"))

        print("\n" + "-" * 80)
        print("Δημιουργία γραφημάτων...")

        print("\n-> Οπτικοποίηση για το Κείμενο 1 (PCA)...")
        visualize_embeddings(texts_to_compare_1, glove_model, method='pca', title_prefix='Κείμενο 1:')

        print("-> Οπτικοποίηση για το Κείμενο 1 (t-SNE)...")
        visualize_embeddings(texts_to_compare_1, glove_model, method='tsne', title_prefix='Κείμενο 1:')

        print("\n-> Οπτικοποίηση για το Κείμενο 2 (PCA)...")
        visualize_embeddings(texts_to_compare_2, glove_model, method='pca', title_prefix='Κείμενο 2:')

        print("-> Οπτικοποίηση για το Κείμενο 2 (t-SNE)...")
        visualize_embeddings(texts_to_compare_2, glove_model, method='tsne', title_prefix='Κείμενο 2:')

        print("\nΗ ανάλυση ολοκληρώθηκε.")