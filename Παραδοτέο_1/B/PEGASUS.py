# Παραδοτέο 1
# Ερώτημα Β
# Μέρος 2: Χρήση του HappyTransformer/PEGASUS για τη διόρθωση των κειμένων

import nltk
from happytransformer import HappyTextToText, TTSettings

def reconstruct_paragraph(text):
    sentences = nltk.sent_tokenize(text)

    print("Αρχικοποίηση μοντέλου PEGASUS...")
    happy_tt = HappyTextToText("PEGASUS", "tuner007/pegasus_paraphrase")
    args = TTSettings(num_beams=5, max_length=60)

    reconstructed_sentences = []
    print(f"\nΒρέθηκαν {len(sentences)} προτάσεις για επεξεργασία.\n")
    for i, sentence in enumerate(sentences):
        result = happy_tt.generate_text(sentence, args=args)
        reconstructed_sentences.append(result.text)
        print(f"[{i+1}/{len(sentences)}] Πρωτότυπο: {sentence}")
        print(f"[{i+1}/{len(sentences)}] Ανακατασκευασμένο: {result.text}\n")


    return " ".join(reconstructed_sentences)

# Αρχικά κείμενα
text_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"""

text_2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"""

print("Ανακατασκευή του κειμένου 1 με το μοντέλο PEGASUS:\n")
final_text_1 = reconstruct_paragraph(text_1)

print("\n" + "-"*42)
print("\nΠΡΩΤΟΤΥΠΟ ΚΕΙΜΕΝΟ 1:\n", text_1)
print("\nΑΝΑΚΑΤΑΣΚΕΥΑΣΜΕΝΟ ΚΕΙΜΕΝΟ 1:\n", final_text_1)

print("Ανακατασκευή του κειμένου 2 με το μοντέλο PEGASUS:\n")
final_text_2 = reconstruct_paragraph(text_2)

print("\n" + "-"*42)

print("\nΠΡΩΤΟΤΥΠΟ ΚΕΙΜΕΝΟ 2:\n", text_2)
print("\nΑΝΑΚΑΤΑΣΚΕΥΑΣΜΕΝΟ ΚΕΙΜΕΝΟ 2:\n", final_text_2)