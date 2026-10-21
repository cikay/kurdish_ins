"""The prompt the instruction-gen stage sends to Gemini."""

SINGLE_SPAN_PROMPT = """You are generating instruction-tuning data for a Kurmanji Kurdish language model.
Each item you produce becomes an (instruction, response) training pair, where the response
is a passage excerpt. The model must learn to give rich, complete, multi-sentence answers,
so short factoid answers (a name, a date, a number, a few words) are useless here.

You are given a text passage in a low-resource language (Kurmanji Kurdish).

Step 1: Translate the passage to English in your head to understand it. Do not output the translation.

Step 2: Write exactly 1 question or instruction in Kurmanji. Rules:
- Make the question AS COMPREHENSIVE AS POSSIBLE: it should cover the main topic of the
  passage and as many of its key points as one coherent question can, so that its answer is
  the largest meaningful part of the passage rather than a single detail.
- The answer must be a continuous span that appears EXACTLY in the passage.
- Make the answer span AS LONG AS POSSIBLE: prefer whole paragraphs or several consecutive
  complete sentences. Never stop an answer in the middle of a sentence.
- Ask an open-ended question that naturally requires a long answer, e.g. explain, describe,
  summarize what is known about X, why, how, what happened, what are the details of X.
  Avoid who/when/how-many questions that are answered by a single word or phrase.
- The question must be fully answered by the span, and the span should not contain much
  unrelated material.
- The question must be SELF-CONTAINED: during training it is shown WITHOUT the passage, so a
  reader who has never seen the passage must know exactly what is being asked.
  - Name every entity explicitly: the specific law and country (e.g. "the Syrian Penal Code",
    not "the law"), the person, organization, place, event and time period.
  - Never refer to the passage itself: no "according to the text/article/news", "in this
    passage", "the author", "the girl", "the town", "article 49" without the law's name, etc.
  - If the passage does not give enough context to make a question self-contained (e.g. it
    quotes "article 49" but never says which law), return an empty list.
- Copy the answer span character for character. Do not translate, shorten, or rephrase it.
- If no question with a long, complete answer exists, return an empty list.

Output valid JSON only: a list with one item, or an empty list:
[
  {"question": "...", "answer_span": "exact text copied from the passage"}
]"""
