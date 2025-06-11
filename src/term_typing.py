import google.generativeai as genai
import os
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_term(term, classes):
    """
    Classify a term using Gemini API with a template.

    Args:
        term (str): The term to classify.
        classes (list): A list of classes to check against.

    Returns:
        str: Mapped class name or "unknown" if no match found.
    """
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Create template with placeholders
    template = """Given the term "<TERM>" and the following WordNet part-of-speech classes: <CLASSES>, classify the term into the most appropriate grammatical type. 
    
    Use WordNet as reference to determine the primary part-of-speech for this term. Consider:
    - noun: entities, objects, concepts, people, places, things
    - verb: actions, processes, states
    - adjective: descriptive properties, qualities, attributes
    - adverb: manner, time, place, degree modifiers
    
    Term: <TERM>
    WordNet POS Classes: <CLASSES>
    
    Please respond with only the most suitable part-of-speech class for this term, without any additional text or explanation."""

    # Fill template placeholders
    classes_str = ", ".join(classes)
    prompt = template.replace("<TERM>", term).replace("<CLASSES>", classes_str)

    # Call Gemini API
    try:
        response = model.generate_content(prompt)
        raw_response = response.text
        
        # Map response to one of the supplied classes
        mapped_class = map_response_to_class(raw_response, classes, term)
        
        return mapped_class
    except Exception as e:
        logger.error(f"Error classifying term '{term}': {str(e)}")
        return "unknown"

def map_response_to_class(response, classes, term):
    """
    Map LLM response to one of the supplied classes via string matching.
    
    Args:
        response (str): Raw response from LLM
        classes (list): List of valid classes
        term (str): Original term for logging
    
    Returns:
        str: Matched class or "unknown"
    """
    # Clean response
    response_lower = response.lower().strip()
    
    # Try exact matching first
    for class_name in classes:
        if class_name.lower() in response_lower:
            logger.info(f"Term '{term}' classified as '{class_name}' (exact match)")
            return class_name
    
    # Try partial matching with word boundaries
    import re
    for class_name in classes:
        pattern = r'\b' + re.escape(class_name.lower()) + r'\b'
        if re.search(pattern, response_lower):
            logger.info(f"Term '{term}' classified as '{class_name}' (word boundary match)")
            return class_name
    
    # No match found
    logger.warning(f"Term '{term}' could not be mapped. Response: '{response}'. Available classes: {classes}")
    return "unknown"

def classify_random_terms(num_terms, classes, terms_file_path="data/terms.txt", ontology_file_path="prolog/ontology.pl"):
    """
    Select random terms from terms file, classify them, and write to Prolog ontology file.
    
    Args:
        num_terms (int): Number of random terms to classify
        classes (list): List of classes to classify against
        terms_file_path (str): Path to the terms file
        ontology_file_path (str): Path to the Prolog ontology file
    
    Returns:
        list: List of tuples containing (term, classified_class)
    """
    # Read all terms from file
    try:
        with open(terms_file_path, 'r') as f:
            all_terms = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Terms file not found: {terms_file_path}")
        return []
    
    # Select random terms
    if num_terms > len(all_terms):
        logger.warning(f"Requested {num_terms} terms, but only {len(all_terms)} available. Using all terms.")
        selected_terms = all_terms
    else:
        selected_terms = random.sample(all_terms, num_terms)
    
    results = []
    
    # Classify each term and write to ontology file
    with open(ontology_file_path, 'a') as ontology_file:
        for term in selected_terms:
            logger.info(f"Classifying term: {term}")
            classified_class = classify_term(term, classes)
            
            # Only write to Prolog file if classification is not 'unknown'
            if classified_class != "unknown":
                prolog_fact = f"type('{term}','{classified_class}').\n"
                ontology_file.write(prolog_fact)
                logger.info(f"Written to ontology: {prolog_fact.strip()}")
            else:
                logger.info(f"Skipped writing '{term}' to ontology (classified as unknown)")
            
            results.append((term, classified_class))
    
    logger.info(f"Successfully classified and wrote {len(results)} terms to {ontology_file_path}")
    return results
