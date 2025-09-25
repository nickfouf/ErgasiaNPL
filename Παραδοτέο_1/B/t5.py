# Παραδοτέο 1
# Ερώτημα Β
# Μέρος 3: Χρήση του HappyTransformer/T5 για τη διόρθωση των κειμένων

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Γίνεται λήψη...")
    nltk.download('punkt')
    print("Η λήψη ολοκληρώθηκε.")


def reconstruct_with_t5_grammar_correction(text: str, model_name: str = "vennify/t5-base-grammar-correction") -> str:
    # Φόρτωση του tokenizer και του μοντέλου
    print("Φόρτωση μοντέλου T5 για διόρθωση γραμματικής...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Το μοντέλο φορτώθηκε επιτυχώς και τρέχει σε: {device}")

    sentences = nltk.sent_tokenize(text)

    reconstructed_sentences = []
    print(f"\nΒρέθηκαν {len(sentences)} προτάσεις για επεξεργασία.\n")
    for i, sentence in enumerate(sentences):
        input_text = f"gec: {sentence}"

        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        outputs = model.generate(
            inputs,
            max_length=150,
            num_beams=5,
            early_stopping=True
        )

        reconstructed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reconstructed_sentences.append(reconstructed_sentence)

        print(f"[{i+1}/{len(sentences)}] Πρωτότυπο: {sentence}")
        print(f"[{i+1}/{len(sentences)}] Ανακατασκευασμένο: {reconstructed_sentence}\n")

    return " ".join(reconstructed_sentences)

# Αρχικό κείμενο
text_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"""

text_2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"""

print("Ανακατασκευή του κειμένου 1 με το μοντέλο T5:\n")
final_text_1 = reconstruct_with_t5_grammar_correction(text_1)

print("-"*42)
print("\nΠΡΩΤΟΤΥΠΟ ΚΕΙΜΕΝΟ 1:\n", text_1)
print("\nΑΝΑΚΑΤΑΣΚΕΥΑΣΜΕΝΟ ΚΕΙΜΕΝΟ 1:\n", final_text_1)

print("Ανακατασκευή του κειμένου 2 με το μοντέλο T5:\n")
final_text_2 = reconstruct_with_t5_grammar_correction(text_2)

print("\n" + "-"*42)

print("\nΠΡΩΤΟΤΥΠΟ ΚΕΙΜΕΝΟ 2:\n", text_2)
print("\nΑΝΑΚΑΤΑΣΚΕΥΑΣΜΕΝΟ ΚΕΙΜΕΝΟ 2:\n", final_text_2)