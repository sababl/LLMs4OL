import os
import random
import time
from nltk.corpus import wordnet as wn
import google.generativeai as genai

from dotenv import load_dotenv


from itertools import islice


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)


def type_term(term: str) -> str:
    """Return the type of a term (noun, verb, adjective, adverb)."""
    prompt = f""" Fill the [MASK] in the following sentence with the correct part of speech. answer in the form of a single word.
            {term} part of speech is a [MASK]."""
    response = model.generate_content(prompt)

    print(f"Term: {term}, Type: {response.text}")
    return response.text


def create_samples(neg_seed, pos_seed, n: int) -> list:
    """Create a list of n random samples from the hypernym chain."""
    syn = wn.synsets(pos_seed, pos=wn.NOUN)[0]
    path = syn.hypernym_paths()[0]
    chain = [s.lemmas()[0].name() for s in path]
    print(f"Hypernym chain for {pos_seed}: {chain}")
    positives = [
        (chain[i], chain[j])
        for i in range(len(chain))
        for j in range(i + 1, len(chain))
    ]
    neg_syn = wn.synsets(neg_seed, pos=wn.NOUN)[0]
    neg_chain = [s.lemmas()[0].name() for s in neg_syn.hypernym_paths()[0][:5]]

    negatives = [(a, b) for a in chain for b in neg_chain]

    num_pos = min(n // 2, len(positives))
    num_neg = min(n - num_pos, len(negatives))

    test_pairs = random.sample(positives, num_pos) + random.sample(negatives, num_neg)
    random.shuffle(test_pairs)
    print(f"Test pairs: {test_pairs}")
    return test_pairs

def is_subclass(test_pairs) -> str:
    """Check if a term is a subclass of another term."""
    responses = []
    for child, parent in test_pairs:
        prompt = f""" Fill the [MASK] in the following sentence with True or False. answer in the form of a single word.
                    {parent} is the superclass of {child}. This statement is [MASK]."""
        response = model.generate_content(prompt)
        responses.append((child, parent, response.text))
        print(f"Term: {child}, Parent: {parent}, Subclass: {response.text}")
        time.sleep(5)  # Avoid rate limiting
    
    return responses
