import os
import random
import time
from nltk.corpus import wordnet as wn
import google.generativeai as genai

from dotenv import load_dotenv


from itertools import combinations, islice


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")


def type_term(term: str) -> str:
    """Return the type of a term (noun, verb, adjective, adverb)."""
    prompt = f"{term} part of speech is a [MASK]."
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
    test_set = positives + negatives
    random.shuffle(test_set)
    print(f"Test set: {test_set}")

    test_pairs = list(islice(positives, 5)) + list(islice(negatives, 5))
    random.shuffle(test_pairs)
    return test_pairs

def is_subclass(test_pairs) -> str:
    """Check if a term is a subclass of another term."""
    responses = []
    for child, parent in test_pairs:
        prompt = f"{parent} is the superclass of {child}. This staement is [MASK]."
        response = model.generate_content(prompt)
        responses.append((child, parent, response.text))
        print(f"Term: {child}, Parent: {parent}, Subclass: {response.text}")
        time.sleep(5)  # Avoid rate limiting
    
    return responses