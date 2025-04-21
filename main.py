import time
from tasks import *
from nltk.corpus import wordnet as wn

if __name__ == "__main__":

    #collect 200 random noun lemmas
    terms = {lem.name().replace('_',' ') for syn in wn.all_synsets('n') 
            for lem in syn.lemmas()}
    terms = list(terms)[:10]
    types = {}
    for t in terms:
        types[t] = type_term(t)
        time.sleep(5) #avoid rate limiting

    test_pairs = create_samples("chair", "dog", 10)
    print(f"Test pairs: {test_pairs}")
    responses = is_subclass(test_pairs)
