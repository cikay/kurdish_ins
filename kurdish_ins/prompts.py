"""The prompt the instruction-gen stage sends to Gemini."""

MULTI_SPAN_PROMPT = """You are generating instruction-tuning data for a Kurmanji Kurdish language model.
Each item you produce becomes an (instruction, response) training pair, where the response
is built from passage excerpts. The model must learn to give rich, complete, multi-sentence
answers, so short factoid answers (a name, a date, a number, a few words) are useless here.

You are given a text passage in a low-resource language (Kurmanji Kurdish).

Step 1: Translate the passage to English in your head to understand it. Do not output the translation.

Step 2: Write exactly 1 question or instruction in Kurmanji. Rules:
- Make the question AS COMPREHENSIVE AS POSSIBLE: it should cover the main topic of the
  passage and as many of its key points as one coherent question can, so that its answer is
  the largest meaningful part of the passage rather than a single detail.
- The answer is made of one or more spans copied EXACTLY from the passage. The spans are
  joined in order to form the final answer, so:
  - List the spans in the SAME ORDER as they appear in the passage. Spans must not overlap.
  - Each span must be one or more complete sentences. Never start or stop a span in the
    middle of a sentence.
  - Use several spans only to skip material that does not belong in the answer (menus, ads,
    captions, unrelated sentences). Prefer few, long spans over many short ones.
  - The joined spans must read as one natural, coherent answer: do not skip a sentence if a
    later span depends on it (e.g. it refers back with "this", "he", "after that").
- Make the answer AS LONG AS POSSIBLE while keeping it relevant to the question.
- Ask an open-ended question that naturally requires a long answer, e.g. explain, describe,
  summarize what is known about X, why, how, what happened, what are the details of X.
  Avoid who/when/how-many questions that are answered by a single word or phrase.
- The question must be fully answered by the joined spans.
- The question must be SELF-CONTAINED: during training it is shown WITHOUT the passage, so a
  reader who has never seen the passage must know exactly what is being asked.
  - Name every entity explicitly: the specific law and country (e.g. "the Syrian Penal Code",
    not "the law"), the person, organization, place, event and time period.
  - Never refer to the passage itself: no "according to the text/article/news", "in this
    passage", "the author", "the girl", "the town", "article 49" without the law's name, etc.
  - If the passage does not give enough context to make a question self-contained (e.g. it
    quotes "article 49" but never says which law), return null.
- Copy every span character for character. Do not translate, shorten, or rephrase it.
- If no question with a long, complete answer exists, return null.

Output valid JSON only: one object, or null:
{"question": "...", "answer_spans": ["first exact span", "second exact span, later in the passage"]}"""
