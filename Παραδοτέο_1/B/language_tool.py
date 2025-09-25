# Παραδοτέο 1
# Ερώτημα Β
# Μέρος 1: Χρήση του LanguageTool για τη διόρθωση των κειμένων

import language_tool_python


text_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

text_2 = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""

print("Διόρθωση κειμένων με το LanguageTool...")

# Αρχικοποίηση του εργαλείου
tool = language_tool_python.LanguageTool('en-US')

# Διόρθωση των κειμένων
corrected_text_1_lt = tool.correct(text_1)
corrected_text_2_lt = tool.correct(text_2)

print("\nΚείμενο 1 - Διορθωμένο:")
print(corrected_text_1_lt)
print("\n" + "-"*100)
print("\nΚείμενο 2 - Διορθωμένο:")
print(corrected_text_2_lt)