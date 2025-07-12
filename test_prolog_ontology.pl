% Test queries for the UMLS Prolog ontology
% This file demonstrates how to use the generated ontology

% Load the UMLS ontology
:- consult('prolog/umls_ontology.pl').

% Test queries
test_ontology :-
    write('=== UMLS Prolog Ontology Test Queries ==='), nl, nl,
    
    % Test 1: Count total concepts
    write('1. Total number of concepts:'), nl,
    findall(X, concept(X, _, _), Concepts),
    length(Concepts, ConceptCount),
    format('   Found ~w concepts', [ConceptCount]), nl, nl,
    
    % Test 2: Find all organisms
    write('2. All organism types:'), nl,
    findall(X, subclass(X, organism), Organisms),
    forall(member(Org, Organisms), 
           (format('   - ~w', [Org]), nl)),
    nl,
    
    % Test 3: Find ancestors of human
    write('3. Ancestors of human:'), nl,
    findall(X, ancestor(X, human), Ancestors),
    forall(member(Anc, Ancestors), 
           (format('   - ~w', [Anc]), nl)),
    nl,
    
    % Test 4: Find siblings of plant
    write('4. Siblings of plant:'), nl,
    findall(X, sibling(plant, X), Siblings),
    forall(member(Sib, Siblings), 
           (format('   - ~w', [Sib]), nl)),
    nl,
    
    % Test 5: Find concepts with definitions
    write('5. Sample concepts with definitions (first 5):'), nl,
    findall(X-Def, (has_definition(X), definition(X, Def)), DefsAll),
    length(DefsAll, DefCount),
    format('   Total concepts with definitions: ~w', [DefCount]), nl,
    take(5, DefsAll, DefsSample),
    forall(member(Concept-Definition, DefsSample),
           (format('   - ~w: ~w', [Concept, Definition]), nl)),
    nl,
    
    % Test 6: Common ancestors
    write('6. Common ancestors of plant and animal:'), nl,
    findall(X, common_ancestor(plant, animal, X), CommonAncs),
    forall(member(CA, CommonAncs), 
           (format('   - ~w', [CA]), nl)),
    nl,
    
    write('Test completed successfully!'), nl.

% Helper predicate to take first N elements
take(0, _, []) :- !.
take(_, [], []) :- !.
take(N, [H|T], [H|R]) :-
    N > 0,
    N1 is N - 1,
    take(N1, T, R).

% Main entry point
:- initialization(test_ontology).
