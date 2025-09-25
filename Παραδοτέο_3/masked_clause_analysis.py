from transformers import pipeline
import spacy
from tabulate import tabulate


def run_masked_clause_analysis():
    """
        μοντέλο 1: GreekBERT (εξειδικευμένο στα Ελληνικά)
        μοντέλο 2: XLM-RoBERTa (πολυγλωσσικό)
    """

    # Φόρτωση Μοντέλων
    print("Φόρτωση μοντέλων...")
    try:
        greek_bert = pipeline("fill-mask", model="nlpaueb/bert-base-greek-uncased-v1")
        xlm_roberta = pipeline("fill-mask", model="xlm-roberta-base")
        nlp = spacy.load("el_core_news_md")
        print("Τα μοντέλα φορτώθηκαν επιτυχώς.")
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση των μοντέλων: {e}")
        return

    # Δεδομένα Εισόδου
    clauses_to_analyze = [
        {
            "article": "Άρθρο 1113",
            "masked_text": "Αν η κυριότητα του [MASK] ανήκει σε περισσότερους.",
            "ground_truth": "πράγματος",
            "full_sentence_truth": "Αν η κυριότητα του πράγματος ανήκει σε περισσότερους.",
        },
        {
            "article": "Άρθρο 1114 - Τίτλος",
            "masked_text": "Πραγματική δουλεία σε [MASK] ή υπέρ του κοινού ακινήτου.",
            "ground_truth": "κοινό",
            "full_sentence_truth": "Πραγματική δουλεία σε κοινό ακίνητο.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Α",
            "masked_text": "Στο κοινό [MASK] μπορεί να συσταθεί πραγματική δουλεία.",
            "ground_truth": "ακίνητο",
            "full_sentence_truth": "Στο κοινό ακίνητο μπορεί να συσταθεί πραγματική δουλεία.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Β",
            "masked_text": "υπέρ του [MASK] κυρίου άλλου ακινήτου.",
            "ground_truth": "εκάστοτε",
            "full_sentence_truth": "υπέρ του εκάστοτε κυρίου άλλου ακινήτου.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Γ",
            "masked_text": "αυτός είναι [MASK] του ακινήτου που βαρύνεται.",
            "ground_truth": "κύριος",
            "full_sentence_truth": "αυτός είναι κύριος του ακινήτου.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Δ",
            "masked_text": "για την [MASK] δουλεία πάνω σε ακίνητο.",
            "ground_truth": "πραγματική",
            "full_sentence_truth": "για την πραγματική δουλεία πάνω σε ακίνητο.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Ε",
            "masked_text": "αν [MASK] από αυτούς είναι κύριος του ακινήτου.",
            "ground_truth": "ένας",
            "full_sentence_truth": "αν ένας από αυτούς είναι κύριος του ακινήτου.",
        },
        {
            "article": "Άρθρο 1114 - Μέρος Ζ",
            "masked_text": "είναι κύριος του [MASK] που βαρύνεται.",
            "ground_truth": "ακινήτου",
            "full_sentence_truth": "είναι κύριος του ακινήτου που βαρύνεται.",
        },
    ]

    # Εκτέλεση Πρόβλεψης και Σύγκριση
    print("\n" + "-" * 80)
    print("Συγκριτική Ανάλυση: GreekBERT vs XLM-RoBERTa")
    print("-" * 80)

    for item in clauses_to_analyze:
        print(f"\n\nΑνάλυση: {item['article']}")
        print(f"Πρόταση: {item['masked_text']}")
        print(f"Σωστή Λέξη (Ground Truth): '{item['ground_truth']}'")
        print("-" * 80)

        # Υπολογισμός προβλέψεων
        try:
            greek_bert_preds = greek_bert(item['masked_text'].replace("[MASK]", greek_bert.tokenizer.mask_token),
                                          top_k=5)
            xlm_roberta_preds = xlm_roberta(item['masked_text'].replace("[MASK]", xlm_roberta.tokenizer.mask_token),
                                            top_k=5)

            table_data = []
            headers = ["Rank", "Πρόβλεψη GreekBERT", "Score", "Πρόβλεψη XLM-RoBERTa", "Score"]

            for i in range(5):
                # GreekBERT data
                gb_pred = greek_bert_preds[i]
                gb_token = gb_pred['token_str'].strip()
                gb_score = gb_pred['score']
                if gb_token == item['ground_truth']:
                    gb_token += " (Σωστό)"

                # XLM-RoBERTa data
                xlm_pred = xlm_roberta_preds[i]
                xlm_token = xlm_pred['token_str'].strip()
                xlm_score = xlm_pred['score']
                if xlm_token == item['ground_truth']:
                    xlm_token += " (Σωστό)"

                table_data.append([i + 1, gb_token, gb_score, xlm_token, xlm_score])

            # Εμφάνιση αποτελεσμάτων σε πίνακα
            print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="github"))

        except Exception as e:
            print(f"Σφάλμα κατά την πρόβλεψη: {e}")

    # Συντακτική Ανάλυση για Γραμματική Αξιολόγηση
    print("\n\n" + "=" * 80)
    print("Συντακτική Ανάλυση με spaCy")
    print("=" * 80)

    for item in clauses_to_analyze:
        print(f"\n Συντακτική Ανάλυση: '{item['full_sentence_truth']}'")
        doc = nlp(item['full_sentence_truth'])

        spacy_table_data = []
        headers = ["Token", "Μέρος του Λόγου", "Συντακτική σχέση", "Κύριο Μέρος"]
        for token in doc:
            spacy_table_data.append([token.text, token.pos_, token.dep_, token.head.text])

        # Εμφάνιση αποτελεσμάτων σε πίνακα
        print(tabulate(spacy_table_data, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    run_masked_clause_analysis()