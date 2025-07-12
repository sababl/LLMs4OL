% UMLS Ontology in Prolog
% Automatically generated from UMLS JSON data
% Generated on: 2025-06-14 17:41:22
%
% This file contains:
% - UMLS semantic type concepts
% - Concept definitions
% - Hierarchical (is-a) relationships
% - Semantic (non-is-a) relationships
% - Useful rules for querying
% - Example queries

:- encoding(utf8).

% UMLS Semantic Type Concepts
% concept(ConceptName, UI, TreeNumber).

concept(organism, 'T001', 'A1.1').
concept(plant, 'T002', 'A1.1.3.3').
concept(fungus, 'T004', 'A1.1.3.2').
concept(virus, 'T005', 'A1.1.4').
concept(bacterium, 'T007', 'A1.1.2').
concept(animal, 'T008', 'A1.1.3.1').
concept(vertebrate, 'T010', 'A1.1.3.1.1').
concept(amphibian, 'T011', 'A1.1.3.1.1.1').
concept(bird, 'T012', 'A1.1.3.1.1.2').
concept(fish, 'T013', 'A1.1.3.1.1.3').
concept(reptile, 'T014', 'A1.1.3.1.1.5').
concept(mammal, 'T015', 'A1.1.3.1.1.4').
concept(human, 'T016', 'A1.1.3.1.1.4.1').
concept(anatomical_structure, 'T017', 'A1.2').
concept(embryonic_structure, 'T018', 'A1.2.1').
concept(congenital_abnormality, 'T019', 'A1.2.2.1').
concept(acquired_abnormality, 'T020', 'A1.2.2.2').
concept(fully_formed_anatomical_structure, 'T021', 'A1.2.3').
concept(body_system, 'T022', 'A2.1.4.1').
concept(body_part_organ_or_organ_component, 'T023', 'A1.2.3.1').
concept(tissue, 'T024', 'A1.2.3.2').
concept(cell, 'T025', 'A1.2.3.3').
concept(cell_component, 'T026', 'A1.2.3.4').
concept(gene_or_genome, 'T028', 'A1.2.3.5').
concept(body_location_or_region, 'T029', 'A2.1.5.2').
concept(body_space_or_junction, 'T030', 'A2.1.5.1').
concept(body_substance, 'T031', 'A1.4.2').
concept(organism_attribute, 'T032', 'A2.3').
concept(finding, 'T033', 'A2.2').
concept(laboratory_or_test_result, 'T034', 'A2.2.1').
concept(injury_or_poisoning, 'T037', 'B2.3').
concept(biologic_function, 'T038', 'B2.2.1').
concept(physiologic_function, 'T039', 'B2.2.1.1').
concept(organism_function, 'T040', 'B2.2.1.1.1').
concept(mental_process, 'T041', 'B2.2.1.1.1.1').
concept(organ_or_tissue_function, 'T042', 'B2.2.1.1.2').
concept(cell_function, 'T043', 'B2.2.1.1.3').
concept(molecular_function, 'T044', 'B2.2.1.1.4').
concept(genetic_function, 'T045', 'B2.2.1.1.4.1').
concept(pathologic_function, 'T046', 'B2.2.1.2').
concept(disease_or_syndrome, 'T047', 'B2.2.1.2.1').
concept(mental_or_behavioral_dysfunction, 'T048', 'B2.2.1.2.1.1').
concept(cell_or_molecular_dysfunction, 'T049', 'B2.2.1.2.2').
concept(experimental_model_of_disease, 'T050', 'B2.2.1.2.3').
concept(event, 'T051', 'B').
concept(activity, 'T052', 'B1').
concept(behavior, 'T053', 'B1.1').
concept(social_behavior, 'T054', 'B1.1.1').
concept(individual_behavior, 'T055', 'B1.1.2').
concept(daily_or_recreational_activity, 'T056', 'B1.2').
concept(occupational_activity, 'T057', 'B1.3').
concept(health_care_activity, 'T058', 'B1.3.1').
concept(laboratory_procedure, 'T059', 'B1.3.1.1').
concept(diagnostic_procedure, 'T060', 'B1.3.1.2').
concept(therapeutic_or_preventive_procedure, 'T061', 'B1.3.1.3').
concept(research_activity, 'T062', 'B1.3.2').
concept(molecular_biology_research_technique, 'T063', 'B1.3.2.1').
concept(governmental_or_regulatory_activity, 'T064', 'B1.3.3').
concept(educational_activity, 'T065', 'B1.3.4').
concept(machine_activity, 'T066', 'B1.4').
concept(phenomenon_or_process, 'T067', 'B2').
concept(human_caused_phenomenon_or_process, 'T068', 'B2.1').
concept(environmental_effect_of_humans, 'T069', 'B2.1.1').
concept(natural_phenomenon_or_process, 'T070', 'B2.2').
concept(entity, 'T071', 'A').
concept(physical_object, 'T072', 'A1').
concept(manufactured_object, 'T073', 'A1.3').
concept(medical_device, 'T074', 'A1.3.1').
concept(research_device, 'T075', 'A1.3.2').
concept(conceptual_entity, 'T077', 'A2').
concept(idea_or_concept, 'T078', 'A2.1').
concept(temporal_concept, 'T079', 'A2.1.1').
concept(qualitative_concept, 'T080', 'A2.1.2').
concept(quantitative_concept, 'T081', 'A2.1.3').
concept(spatial_concept, 'T082', 'A2.1.5').
concept(geographic_area, 'T083', 'A2.1.5.4').
concept(molecular_sequence, 'T085', 'A2.1.5.3').
concept(nucleotide_sequence, 'T086', 'A2.1.5.3.1').
concept(amino_acid_sequence, 'T087', 'A2.1.5.3.2').
concept(carbohydrate_sequence, 'T088', 'A2.1.5.3.3').
concept(regulation_or_law, 'T089', 'A2.4.2').
concept(occupation_or_discipline, 'T090', 'A2.6').
concept(biomedical_occupation_or_discipline, 'T091', 'A2.6.1').
concept(organization, 'T092', 'A2.7').
concept(health_care_related_organization, 'T093', 'A2.7.1').
concept(professional_society, 'T094', 'A2.7.2').
concept(self_help_or_relief_organization, 'T095', 'A2.7.3').
concept(group, 'T096', 'A2.9').
concept(professional_or_occupational_group, 'T097', 'A2.9.1').
concept(population_group, 'T098', 'A2.9.2').
concept(family_group, 'T099', 'A2.9.3').
concept(age_group, 'T100', 'A2.9.4').
concept(patient_or_disabled_group, 'T101', 'A2.9.5').
concept(group_attribute, 'T102', 'A2.8').
concept(chemical, 'T103', 'A1.4.1').
concept(chemical_viewed_structurally, 'T104', 'A1.4.1.2').
concept(organic_chemical, 'T109', 'A1.4.1.2.1').
concept(nucleic_acid_nucleoside_or_nucleotide, 'T114', 'A1.4.1.2.1.5').
concept(amino_acid_peptide_or_protein, 'T116', 'A1.4.1.2.1.7').
concept(chemical_viewed_functionally, 'T120', 'A1.4.1.1').
concept(pharmacologic_substance, 'T121', 'A1.4.1.1.1').
concept(biomedical_or_dental_material, 'T122', 'A1.4.1.1.2').
concept(biologically_active_substance, 'T123', 'A1.4.1.1.3').
concept(hormone, 'T125', 'A1.4.1.1.3.2').
concept(enzyme, 'T126', 'A1.4.1.1.3.3').
concept(vitamin, 'T127', 'A1.4.1.1.3.4').
concept(immunologic_factor, 'T129', 'A1.4.1.1.3.5').
concept(indicator_reagent_or_diagnostic_aid, 'T130', 'A1.4.1.1.4').
concept(hazardous_or_poisonous_substance, 'T131', 'A1.4.1.1.5').
concept(substance, 'T167', 'A1.4').
concept(food, 'T168', 'A1.4.3').
concept(functional_concept, 'T169', 'A2.1.4').
concept(intellectual_product, 'T170', 'A2.4').
concept(language, 'T171', 'A2.5').
concept(sign_or_symptom, 'T184', 'A2.2.2').
concept(classification, 'T185', 'A2.4.1').
concept(anatomical_abnormality, 'T190', 'A1.2.2').
concept(neoplastic_process, 'T191', 'B2.2.1.2.1.2').
concept(receptor, 'T192', 'A1.4.1.1.3.6').
concept(archaeon, 'T194', 'A1.1.1').
concept(antibiotic, 'T195', 'A1.4.1.1.1.1').
concept(element_ion_or_isotope, 'T196', 'A1.4.1.2.3').
concept(inorganic_chemical, 'T197', 'A1.4.1.2.2').
concept(clinical_drug, 'T200', 'A1.3.3').
concept(clinical_attribute, 'T201', 'A2.3.1').
concept(drug_delivery_device, 'T203', 'A1.3.1.1').
concept(eukaryote, 'T204', 'A1.1.3').
concept(unknown, '', '').
concept(process_of, '', '').
concept(disrupts, '', '').
concept(produces, '', '').
concept(traverses, '', '').
concept(branch_of, '', '').
concept(conceptual_part_of, '', '').
concept(associated_with, '', '').
concept(connected_to, '', '').
concept(functionally_related_to, '', '').
concept(analyzes, '', '').
concept(manages, '', '').
concept(precedes, '', '').
concept(part_of, '', '').
concept(spatially_related_to, '', '').
concept(degree_of, '', '').
concept(location_of, '', '').
concept(assesses_effect_of, '', '').
concept(prevents, '', '').
concept(causes, '', '').
concept(measures, '', '').
concept(manifestation_of, '', '').
concept(uses, '', '').
concept(result_of, '', '').
concept(tributary_of, '', '').
concept(co_occurs_with, '', '').
concept(developmental_form_of, '', '').
concept(interconnects, '', '').
concept(physically_related_to, '', '').
concept(contains, '', '').
concept(performs, '', '').
concept(surrounds, '', '').
concept(complicates, '', '').
concept(conceptually_related_to, '', '').
concept(method_of, '', '').
concept(practices, '', '').
concept(occurs_in, '', '').
concept(exhibits, '', '').
concept(interacts_with, '', '').
concept(issue_in, '', '').
concept(treats, '', '').
concept(property_of, '', '').
concept(measurement_of, '', '').
concept(temporally_related_to, '', '').
concept(isa, '', '').
concept(carries_out, '', '').
concept(adjacent_to, '', '').
concept(diagnoses, '', '').
concept(derivative_of, '', '').
concept(consists_of, '', '').
concept(evaluation_of, '', '').
concept(affects, '', '').
concept(indicates, '', '').
concept(brings_about, '', '').
concept(ingredient_of, '', '').

% UMLS Concept Definitions
% definition(ConceptName, Definition).

definition(organism, 'Generally, a living individual, including all plants and animals.').
definition(plant, 'An organism having cellulose cell walls, growing by synthesis of inorganic substances, generally distinguished by the presence of chlorophyll, and lacking the power of locomotion. Plant parts are included here as well.').
definition(fungus, 'A eukaryotic organism characterized by the absence of chlorophyll and the presence of a rigid cell wall. Included here are both slime molds and true fungi such as yeasts, molds, mildews, and mushrooms.').
definition(virus, 'An organism consisting of a core of a single nucleic acid enclosed in a protective coat of protein. A virus may replicate only inside a host living cell. A virus exhibits some but not all of the usual characteristics of living things.').
definition(bacterium, 'A small, typically one-celled, prokaryotic micro-organism.').
definition(animal, 'An organism with eukaryotic cells, and lacking stiff cell walls, plastids and photosynthetic pigments.').
definition(vertebrate, 'An animal which has a spinal column.').
definition(amphibian, 'A cold-blooded, smooth-skinned vertebrate which characteristically hatches as an aquatic larva, breathing by gills. When mature, the amphibian breathes with lungs.').
definition(bird, 'A vertebrate having a constant body temperature and characterized by the presence of feathers.').
definition(fish, 'A cold-blooded aquatic vertebrate characterized by fins and breathing by gills. Included here are fishes having either a bony skeleton, such as a perch, or a cartilaginous skeleton, such as a shark, or those lacking a jaw, such as a lamprey or hagfish.').
definition(reptile, 'A cold-blooded vertebrate having an external covering of scales or horny plates. Reptiles breathe by means of lungs and are generally egg-laying.').
definition(mammal, 'A vertebrate having a constant body temperature and characterized by the presence of hair, mammary glands and sweat glands.').
definition(human, 'Modern man, the only remaining species of the Homo genus.').
definition(anatomical_structure, 'A normal or pathological part of the anatomy or structural organization of an organism.').
definition(embryonic_structure, 'An anatomical structure that exists only before the organism is fully formed; in mammals, for example, a structure that exists only prior to the birth of the organism. This structure may be normal or abnormal.').
definition(congenital_abnormality, 'An abnormal structure, or one that is abnormal in size or location, present at birth or evolving over time as a result of a defect in embryogenesis.').
definition(acquired_abnormality, 'An abnormal structure, or one that is abnormal in size or location, found in or deriving from a previously normal structure. Acquired abnormalities are distinguished from diseases even though they may result in pathological functioning (e.g., "hernias incarcerate").').
definition(fully_formed_anatomical_structure, 'An anatomical structure in a fully formed organism; in mammals, for example, a structure in the body after the birth of the organism.').
definition(body_system, 'A complex of anatomical structures that performs a common function.').
definition(body_part_organ_or_organ_component, 'A collection of cells and tissues which are localized to a specific area or combine and carry out one or more specialized functions of an organism. This ranges from gross structures to small components of complex organs. These structures are relatively localized in comparison to tissues.').
definition(tissue, 'An aggregation of similarly specialized cells and the associated intercellular substance. Tissues are relatively non-localized in comparison to body parts, organs or organ components.').
definition(cell, 'The fundamental structural and functional unit of living organisms.').
definition(cell_component, 'A part of a cell or the intercellular matrix, generally visible by light microscopy.').
definition(gene_or_genome, 'A specific sequence, or in the case of the genome the complete sequence, of nucleotides along a molecule of DNA or RNA (in the case of some viruses) which represent the functional units of heredity.').
definition(body_location_or_region, 'An area, subdivision, or region of the body demarcated for the purpose of topographical description.').
definition(body_space_or_junction, 'An area enclosed or surrounded by body parts or organs or the place where two anatomical structures meet or connect.').
definition(body_substance, 'Extracellular material, or mixtures of cells and extracellular material, produced, excreted, or accreted by the body. Included here are substances such as saliva, dental enamel, sweat, and gastric acid.').
definition(organism_attribute, 'A property of the organism or its major parts.').
definition(finding, 'That which is discovered by direct observation or measurement of an organism attribute or condition, including the clinical history of the patient. The history of the presence of a disease is a ''Finding'' and is distinguished from the disease itself.').
definition(laboratory_or_test_result, 'The outcome of a specific test to measure an attribute or to determine the presence, absence, or degree of a condition.').
definition(injury_or_poisoning, 'A traumatic wound, injury, or poisoning caused by an external agent or force.').
definition(biologic_function, 'A state, activity or process of the body or one of its systems or parts.').
definition(physiologic_function, 'A normal process, activity, or state of the body.').
definition(organism_function, 'A physiologic function of the organism as a whole, of multiple organ systems, or of multiple organs or tissues.').
definition(mental_process, 'A physiologic function involving the mind or cognitive processing.').
definition(organ_or_tissue_function, 'A physiologic function of a particular organ, organ system, or tissue.').
definition(cell_function, 'A physiologic function inherent to cells or cell components.').
definition(molecular_function, 'A physiologic function occurring at the molecular level.').
definition(genetic_function, 'Functions of or related to the maintenance, translation or expression of the genetic material.').
definition(pathologic_function, 'A disordered process, activity, or state of the organism as a whole, of a body system or systems, or of multiple organs or tissues. Included here are normal responses to a negative stimulus as well as patholologic conditions or states that are less specific than a disease. Pathologic functions frequently have systemic effects.').
definition(disease_or_syndrome, 'A condition which alters or interferes with a normal process, state, or activity of an organism. It is usually characterized by the abnormal functioning of one or more of the host''s systems, parts, or organs. Included here is a complex of symptoms descriptive of a disorder.').
definition(mental_or_behavioral_dysfunction, 'A clinically significant dysfunction whose major manifestation is behavioral or psychological. These dysfunctions may have identified or presumed biological etiologies or manifestations.').
definition(cell_or_molecular_dysfunction, 'A pathologic function inherent to cells, parts of cells, or molecules.').
definition(experimental_model_of_disease, 'A representation in a non-human organism of a human disease for the purpose of research into its mechanism or treatment.').
definition(event, 'A broad type for grouping activities, processes and states.').
definition(activity, 'An operation or series of operations that an organism or machine carries out or participates in.').
definition(behavior, 'Any of the psycho-social activities of humans or animals that can be observed directly by others or can be made systematically observable by the use of special strategies.').
definition(social_behavior, 'Behavior that is a direct result or function of the interaction of humans or animals with their fellows. This includes behavior that may be considered anti-social.').
definition(individual_behavior, 'Behavior exhibited by a human or an animal that is not a direct result of interaction with other members of the species, but which may have an effect on others.').
definition(daily_or_recreational_activity, 'An activity carried out for recreation or exercise, or as part of daily life.').
definition(occupational_activity, 'An activity carried out as part of an occupation or job.').
definition(health_care_activity, 'An activity of or relating to the practice of medicine or involving the care of patients.').
definition(laboratory_procedure, 'A procedure, method, or technique used to determine the composition, quantity, or concentration of a specimen, and which is carried out in a clinical laboratory. Included here are procedures which measure the times and rates of reactions.').
definition(diagnostic_procedure, 'A procedure, method, or technique used to determine the nature or identity of a disease or disorder. This excludes procedures which are primarily carried out on specimens in a laboratory.').
definition(therapeutic_or_preventive_procedure, 'A procedure, method, or technique designed to prevent a disease or a disorder, or to improve physical function, or used in the process of treating a disease or injury.').
definition(research_activity, 'An activity carried out as part of research or experimentation.').
definition(molecular_biology_research_technique, 'Any of the techniques used in the study of or the directed modification of the gene complement of a living organism.').
definition(governmental_or_regulatory_activity, 'An activity carried out by officially constituted governments, or an activity related to the creation or enforcement of the rules or regulations governing some field of endeavor.').
definition(educational_activity, 'An activity related to the organization and provision of education.').
definition(machine_activity, 'An activity carried out primarily or exclusively by machines.').
definition(phenomenon_or_process, 'A process or state which occurs naturally or as a result of an activity.').
definition(human_caused_phenomenon_or_process, 'A phenomenon or process that is a result of the activities of human beings.').
definition(environmental_effect_of_humans, 'A change in the natural environment that is a result of the activities of human beings.').
definition(natural_phenomenon_or_process, 'A phenomenon or process that occurs irrespective of the activities of human beings.').
definition(entity, 'A broad type for grouping physical and conceptual entities.').
definition(physical_object, 'An object perceptible to the sense of vision or touch.').
definition(manufactured_object, 'A physical object made by human beings.').
definition(medical_device, 'A manufactured object used primarily in the diagnosis, treatment, or prevention of physiologic or anatomic disorders.').
definition(research_device, 'A manufactured object used primarily in carrying out scientific research or experimentation.').
definition(conceptual_entity, 'A broad type for grouping abstract entities or concepts.').
definition(idea_or_concept, 'An abstract concept, such as a social, religious or philosophical concept.').
definition(temporal_concept, 'A concept which pertains to time or duration.').
definition(qualitative_concept, 'A concept which is an assessment of some quality, rather than a direct measurement.').
definition(quantitative_concept, 'A concept which involves the dimensions, quantity or capacity of something using some unit of measure, or which involves the quantitative comparison of entities.').
definition(spatial_concept, 'A location, region, or space, generally having definite boundaries.').
definition(geographic_area, 'A geographic location, generally having definite boundaries.').
definition(molecular_sequence, 'A broad type for grouping the collected sequences of amino acids, carbohydrates, and nucleotide sequences. Descriptions of these sequences are generally reported in the published literature and/or are deposited in and maintained by databanks such as GenBank, European Molecular Biology Laboratory (EMBL), National Biomedical Research Foundation (NBRF), or other sequence repositories.').
definition(nucleotide_sequence, 'The sequence of purines and pyrimidines in nucleic acids and polynucleotides. Included here are nucleotide-rich regions, conserved sequence, and DNA transforming region.').
definition(amino_acid_sequence, 'The sequence of amino acids as arrayed in chains, sheets, etc., within the protein molecule. It is of fundamental importance in determining protein structure.').
definition(carbohydrate_sequence, 'The sequence of carbohydrates within polysaccharides, glycoproteins, and glycolipids.').
definition(regulation_or_law, 'An intellectual product resulting from legislative or regulatory activity.').
definition(occupation_or_discipline, 'A vocation, academic discipline, or field of study, or a subpart of an occupation or discipline.').
definition(biomedical_occupation_or_discipline, 'A vocation, academic discipline, or field of study related to biomedicine.').
definition(organization, 'The result of uniting for a common purpose or function. The continued existence of an organization is not dependent on any of its members, its location, or particular facility. Components or subparts of organizations are also included here. Although the names of organizations are sometimes used to refer to the buildings in which they reside, they are not inherently physical in nature.').
definition(health_care_related_organization, 'An established organization which carries out specific functions related to health care delivery or research in the life sciences.').
definition(professional_society, 'An organization uniting those who have a common vocation or who are involved with a common field of study.').
definition(self_help_or_relief_organization, 'An organization whose purpose and function is to provide assistance to the needy or to offer support to those sharing similar problems.').
definition(group, 'A conceptual entity referring to the classification of individuals according to certain shared characteristics.').
definition(professional_or_occupational_group, 'An individual or individuals classified according to their vocation.').
definition(population_group, 'An indivdual or individuals classified according to their sex, racial origin, religion, common place of living, financial or social status, or some other cultural or behavioral attribute.').
definition(family_group, 'An individual or individuals classified according to their family relationships or relative position in the family unit.').
definition(age_group, 'An individual or individuals classified according to their age.').
definition(patient_or_disabled_group, 'An individual or individuals classified according to a disability, disease, condition or treatment.').
definition(group_attribute, 'A conceptual entity which refers to the frequency or distribution of certain characteristics or phenomena in certain groups.').
definition(chemical, 'Compounds or substances of definite molecular composition. Chemicals are viewed from two distinct perspectives in the network, functionally and structurally. Almost every chemical concept is assigned at least two types, generally one from the structure hierarchy and at least one from the function hierarchy.').
definition(chemical_viewed_structurally, 'A chemical or chemicals viewed from the perspective of their structural characteristics. Included here are concepts which can mean either a salt, an ion, or a compound (e.g., "Bromates" and "Bromides").').
definition(organic_chemical, 'The general class of carbon-containing compounds, usually based on carbon chains or rings, and also containing hydrogen (hydrocarbons), with or without nitrogen, oxygen, or other elements in which the bonding between elements is generally covalent.').
definition(nucleic_acid_nucleoside_or_nucleotide, 'A complex compound of high molecular weight occurring in living cells. These are basically of two types, ribonucleic (RNA) and deoxyribonucleic (DNA) acids. Nucleic acids are made of nucleotides (nitrogen-containing base, a 5-carbon sugar, and one or more phosphate group) linked together by a phosphodiester bond between the 5'' and 3'' carbon atoms. Nucleosides are compounds composed of a purine or pyrimidine base (usually adenine, cytosine, guanine, thymine, uracil) linked to either a ribose or a deoxyribose sugar.').
definition(amino_acid_peptide_or_protein, 'Amino acids and chains of amino acids connected by peptide linkages.').
definition(chemical_viewed_functionally, 'A chemical viewed from the perspective of its functional characteristics or pharmacological activities.').
definition(pharmacologic_substance, 'A substance used in the treatment or prevention of pathologic disorders. This includes substances that occur naturally in the body and are administered therapeutically.').
definition(biomedical_or_dental_material, 'A substance used in biomedicine or dentistry predominantly for its physical, as opposed to chemical, properties. Included here are biocompatible materials, tissue adhesives, bone cements, resins, toothpastes, etc.').
definition(biologically_active_substance, 'A generally endogenous substance produced or required by an organism, of primary interest because of its role in the biologic functioning of the organism that produces it.').
definition(hormone, 'In animals, a chemical usually secreted by an endocrine gland whose products are released into the circulating fluid. Hormones act as chemical messengers and regulate various physiologic processes such as growth, reproduction, metabolism, etc. They usually fall into two broad classes, steroid hormones and peptide hormones.').
definition(enzyme, 'A complex chemical, usually a protein, that is produced by living cells and which catalyzes specific biochemical reactions. There are six main types of enzymes: oxidoreductases, transferases, hydrolases, lyases, isomerases, and ligases.').
definition(vitamin, 'A substance, usually an organic chemical complex, present in natural products or made synthetically, which is essential in the diet of man or other higher animals. Included here are vitamin precursors, provitamins, and vitamin supplements.').
definition(immunologic_factor, 'A biologically active substance whose activities affect or play a role in the functioning of the immune system.').
definition(indicator_reagent_or_diagnostic_aid, 'A substance primarily of interest for its use in laboratory or diagnostic tests and procedures to detect, measure, examine, or analyze other chemicals, processes, or conditions.').
definition(hazardous_or_poisonous_substance, 'A substance of concern because of its potentially hazardous or toxic effects. This would include most drugs of abuse, as well as agents that require special handling because of their toxicity.').
definition(substance, 'A material with definite or fairly definite chemical composition.').
definition(food, 'Any substance generally containing nutrients, such as carbohydrates, proteins, and fats, that can be ingested by a living organism and metabolized into energy and body tissue. Some foods are naturally occurring, others are either partially or entirely made by humans.').
definition(functional_concept, 'A concept which is of interest because it pertains to the carrying out of a process or activity.').
definition(intellectual_product, 'A conceptual entity resulting from human endeavor. Concepts assigned to this type generally refer to information created by humans for some purpose.').
definition(language, 'The system of communication used by a particular nation or people.').
definition(sign_or_symptom, 'An observable manifestation of a disease or condition based on clinical judgment, or a manifestation of a disease or condition which is experienced by the patient and reported as a subjective observation.').
definition(classification, 'A term or system of terms denoting an arrangement by class or category.').
definition(anatomical_abnormality, 'An abnormal structure, or one that is abnormal in size or location.').
definition(neoplastic_process, 'A new and abnormal growth of tissue in which the growth is uncontrolled and progressive. The growths may be malignant or benign.').
definition(receptor, 'A specific structure or site on the cell surface or within its cytoplasm that recognizes and binds with other specific molecules. These include the proteins on the surface of an immunocompetent cell that binds with antigens, or proteins found on the surface molecules that bind with hormones or neurotransmitters and react with other molecules that respond in a specific way.').
definition(archaeon, 'A member of one of the three domains of life, formerly called Archaebacteria under the taxon Bacteria, but now considered separate and distinct. Archaea are characterized by: 1) the presence of characteristic tRNAs and ribosomal RNAs; 2) the absence of peptidoglycan cell walls; 3) the presence of ether-linked lipids built from branched-chain subunits; and 4) their occurrence in unusual habitats. While archaea resemble bacteria in morphology and genomic organization, they resemble eukarya in their method of genomic replication.').
definition(antibiotic, 'A pharmacologically active compound produced by growing microorganisms which kill or inhibit growth of other microorganisms.').
definition(element_ion_or_isotope, 'One of the 109 presently known fundamental substances that comprise all matter at and above the atomic level. This includes elemental metals, rare gases, and most abundant naturally occurring radioactive elements, as well as the ionic counterparts of elements (NA+, Cl-), and the less abundant isotopic forms. This does not include organic ions such as iodoacetate to which the type ''Organic Chemical'' is assigned.').
definition(inorganic_chemical, 'Chemical elements and their compounds, excluding the hydrocarbons and their derivatives (except carbides, carbonates, cyanides, cyanates and carbon disulfide). Generally inorganic compounds contain ionic bonds. Included here are inorganic acids and salts, alloys, alkalies, and minerals.').
definition(clinical_drug, 'A pharmaceutical preparation as produced by the manufacturer. The name usually includes the substance, its strength, and the form, but may include the substance and only one of the other two items.').
definition(clinical_attribute, 'An observable or measurable property or state of an organism of clinical interest.').
definition(drug_delivery_device, 'A medical device that contains a clinical drug or drugs.').
definition(eukaryote, 'One of the three domains of life (the others being Bacteria and Archaea), also called Eukarya. These are organisms whose cells are enclosed in membranes and possess a nucleus. They comprise almost all multicellular and many unicellular organisms, and are traditionally divided into groups (sometimes called kingdoms) including Animals, Plants, Fungi, various Algae, and other taxa that were previously part of the old kingdom Protista.').

% UMLS Hierarchical Relationships (is-a)
% isa(Child, Parent) - Child is a subclass of Parent.

isa(acquired_abnormality, anatomical_abnormality).
isa(activity, event).
isa(age_group, group).
isa(amino_acid_sequence, molecular_sequence).
isa(amino_acid_peptide_or_protein, organic_chemical).
isa(amphibian, vertebrate).
isa(anatomical_abnormality, anatomical_structure).
isa(anatomical_structure, physical_object).
isa(animal, eukaryote).
isa(antibiotic, pharmacologic_substance).
isa(archaeon, organism).
isa(bacterium, organism).
isa(behavior, activity).
isa(biologic_function, natural_phenomenon_or_process).
isa(biologically_active_substance, chemical_viewed_functionally).
isa(biomedical_occupation_or_discipline, occupation_or_discipline).
isa(biomedical_or_dental_material, chemical_viewed_functionally).
isa(bird, vertebrate).
isa(body_location_or_region, spatial_concept).
isa(body_part_organ_or_organ_component, fully_formed_anatomical_structure).
isa(body_space_or_junction, spatial_concept).
isa(body_substance, substance).
isa(body_system, functional_concept).
isa(carbohydrate_sequence, molecular_sequence).
isa(cell_component, fully_formed_anatomical_structure).
isa(cell_function, physiologic_function).
isa(cell_or_molecular_dysfunction, pathologic_function).
isa(cell, fully_formed_anatomical_structure).
isa(chemical_viewed_functionally, chemical).
isa(chemical_viewed_structurally, chemical).
isa(chemical, substance).
isa(classification, intellectual_product).
isa(clinical_attribute, organism_attribute).
isa(clinical_drug, manufactured_object).
isa(conceptual_entity, entity).
isa(congenital_abnormality, anatomical_abnormality).
isa(daily_or_recreational_activity, activity).
isa(diagnostic_procedure, health_care_activity).
isa(disease_or_syndrome, pathologic_function).
isa(drug_delivery_device, medical_device).
isa(educational_activity, occupational_activity).
isa(element_ion_or_isotope, chemical_viewed_structurally).
isa(embryonic_structure, anatomical_structure).
isa(entity, unknown).
isa(environmental_effect_of_humans, human_caused_phenomenon_or_process).
isa(enzyme, biologically_active_substance).
isa(eukaryote, organism).
isa(event, unknown).
isa(experimental_model_of_disease, pathologic_function).
isa(family_group, group).
isa(finding, conceptual_entity).
isa(fish, vertebrate).
isa(food, substance).
isa(fully_formed_anatomical_structure, anatomical_structure).
isa(functional_concept, idea_or_concept).
isa(fungus, eukaryote).
isa(gene_or_genome, fully_formed_anatomical_structure).
isa(genetic_function, molecular_function).
isa(geographic_area, spatial_concept).
isa(governmental_or_regulatory_activity, occupational_activity).
isa(group_attribute, conceptual_entity).
isa(group, conceptual_entity).
isa(hazardous_or_poisonous_substance, chemical_viewed_functionally).
isa(health_care_activity, occupational_activity).
isa(health_care_related_organization, organization).
isa(hormone, biologically_active_substance).
isa(human_caused_phenomenon_or_process, phenomenon_or_process).
isa(human, mammal).
isa(idea_or_concept, conceptual_entity).
isa(immunologic_factor, biologically_active_substance).
isa(indicator_reagent_or_diagnostic_aid, chemical_viewed_functionally).
isa(individual_behavior, behavior).
isa(injury_or_poisoning, phenomenon_or_process).
isa(inorganic_chemical, chemical_viewed_structurally).
isa(intellectual_product, conceptual_entity).
isa(laboratory_procedure, health_care_activity).
isa(laboratory_or_test_result, finding).
isa(language, conceptual_entity).
isa(machine_activity, activity).
isa(mammal, vertebrate).
isa(manufactured_object, physical_object).
isa(medical_device, manufactured_object).
isa(mental_process, organism_function).
isa(mental_or_behavioral_dysfunction, disease_or_syndrome).
isa(molecular_biology_research_technique, research_activity).
isa(molecular_function, physiologic_function).
isa(molecular_sequence, spatial_concept).
isa(natural_phenomenon_or_process, phenomenon_or_process).
isa(neoplastic_process, disease_or_syndrome).
isa(nucleic_acid_nucleoside_or_nucleotide, organic_chemical).
isa(nucleotide_sequence, molecular_sequence).
isa(occupation_or_discipline, conceptual_entity).
isa(occupational_activity, activity).
isa(organ_or_tissue_function, physiologic_function).
isa(organic_chemical, chemical_viewed_structurally).
isa(organism_attribute, conceptual_entity).
isa(organism_function, physiologic_function).
isa(organism, physical_object).
isa(organization, conceptual_entity).
isa(pathologic_function, biologic_function).
isa(patient_or_disabled_group, group).
isa(pharmacologic_substance, chemical_viewed_functionally).
isa(phenomenon_or_process, event).
isa(physical_object, entity).
isa(physiologic_function, biologic_function).
isa(plant, eukaryote).
isa(population_group, group).
isa(professional_society, organization).
isa(professional_or_occupational_group, group).
isa(qualitative_concept, idea_or_concept).
isa(quantitative_concept, idea_or_concept).
isa(receptor, biologically_active_substance).
isa(regulation_or_law, intellectual_product).
isa(reptile, vertebrate).
isa(research_activity, occupational_activity).
isa(research_device, manufactured_object).
isa(self_help_or_relief_organization, organization).
isa(sign_or_symptom, finding).
isa(social_behavior, behavior).
isa(spatial_concept, idea_or_concept).
isa(substance, physical_object).
isa(temporal_concept, idea_or_concept).
isa(therapeutic_or_preventive_procedure, health_care_activity).
isa(tissue, fully_formed_anatomical_structure).
isa(vertebrate, animal).
isa(virus, organism).
isa(vitamin, biologically_active_substance).
isa(adjacent_to, spatially_related_to).
isa(affects, functionally_related_to).
isa(analyzes, conceptually_related_to).
isa(assesses_effect_of, analyzes).
isa(associated_with, unknown).
isa(branch_of, physically_related_to).
isa(brings_about, functionally_related_to).
isa(carries_out, performs).
isa(causes, brings_about).
isa(co_occurs_with, temporally_related_to).
isa(complicates, affects).
isa(conceptual_part_of, conceptually_related_to).
isa(conceptually_related_to, associated_with).
isa(connected_to, physically_related_to).
isa(consists_of, physically_related_to).
isa(contains, physically_related_to).
isa(degree_of, conceptually_related_to).
isa(derivative_of, conceptually_related_to).
isa(developmental_form_of, conceptually_related_to).
isa(diagnoses, conceptually_related_to).
isa(disrupts, affects).
isa(evaluation_of, conceptually_related_to).
isa(exhibits, performs).
isa(functionally_related_to, associated_with).
isa(indicates, functionally_related_to).
isa(ingredient_of, physically_related_to).
isa(interacts_with, affects).
isa(interconnects, physically_related_to).
isa(isa, unknown).
isa(issue_in, conceptually_related_to).
isa(location_of, spatially_related_to).
isa(manages, affects).
isa(manifestation_of, functionally_related_to).
isa(measurement_of, conceptually_related_to).
isa(measures, conceptually_related_to).
isa(method_of, conceptually_related_to).
isa(occurs_in, functionally_related_to).
isa(part_of, physically_related_to).
isa(performs, functionally_related_to).
isa(physically_related_to, associated_with).
isa(practices, performs).
isa(precedes, temporally_related_to).
isa(prevents, affects).
isa(process_of, occurs_in).
isa(produces, brings_about).
isa(property_of, conceptually_related_to).
isa(result_of, functionally_related_to).
isa(spatially_related_to, associated_with).
isa(surrounds, spatially_related_to).
isa(temporally_related_to, associated_with).
isa(traverses, spatially_related_to).
isa(treats, affects).
isa(tributary_of, physically_related_to).
isa(uses, functionally_related_to).

% UMLS Semantic Relationships (non-is-a)
% semantic_relation(Subject, Relation, Object).


% UMLS Concept Properties
% has_ui(ConceptName, UI).
% has_tree_number(ConceptName, TreeNumber).

has_ui(organism, 'T001').
has_tree_number(organism, 'A1.1').
has_ui(plant, 'T002').
has_tree_number(plant, 'A1.1.3.3').
has_ui(fungus, 'T004').
has_tree_number(fungus, 'A1.1.3.2').
has_ui(virus, 'T005').
has_tree_number(virus, 'A1.1.4').
has_ui(bacterium, 'T007').
has_tree_number(bacterium, 'A1.1.2').
has_ui(animal, 'T008').
has_tree_number(animal, 'A1.1.3.1').
has_ui(vertebrate, 'T010').
has_tree_number(vertebrate, 'A1.1.3.1.1').
has_ui(amphibian, 'T011').
has_tree_number(amphibian, 'A1.1.3.1.1.1').
has_ui(bird, 'T012').
has_tree_number(bird, 'A1.1.3.1.1.2').
has_ui(fish, 'T013').
has_tree_number(fish, 'A1.1.3.1.1.3').
has_ui(reptile, 'T014').
has_tree_number(reptile, 'A1.1.3.1.1.5').
has_ui(mammal, 'T015').
has_tree_number(mammal, 'A1.1.3.1.1.4').
has_ui(human, 'T016').
has_tree_number(human, 'A1.1.3.1.1.4.1').
has_ui(anatomical_structure, 'T017').
has_tree_number(anatomical_structure, 'A1.2').
has_ui(embryonic_structure, 'T018').
has_tree_number(embryonic_structure, 'A1.2.1').
has_ui(congenital_abnormality, 'T019').
has_tree_number(congenital_abnormality, 'A1.2.2.1').
has_ui(acquired_abnormality, 'T020').
has_tree_number(acquired_abnormality, 'A1.2.2.2').
has_ui(fully_formed_anatomical_structure, 'T021').
has_tree_number(fully_formed_anatomical_structure, 'A1.2.3').
has_ui(body_system, 'T022').
has_tree_number(body_system, 'A2.1.4.1').
has_ui(body_part_organ_or_organ_component, 'T023').
has_tree_number(body_part_organ_or_organ_component, 'A1.2.3.1').
has_ui(tissue, 'T024').
has_tree_number(tissue, 'A1.2.3.2').
has_ui(cell, 'T025').
has_tree_number(cell, 'A1.2.3.3').
has_ui(cell_component, 'T026').
has_tree_number(cell_component, 'A1.2.3.4').
has_ui(gene_or_genome, 'T028').
has_tree_number(gene_or_genome, 'A1.2.3.5').
has_ui(body_location_or_region, 'T029').
has_tree_number(body_location_or_region, 'A2.1.5.2').
has_ui(body_space_or_junction, 'T030').
has_tree_number(body_space_or_junction, 'A2.1.5.1').
has_ui(body_substance, 'T031').
has_tree_number(body_substance, 'A1.4.2').
has_ui(organism_attribute, 'T032').
has_tree_number(organism_attribute, 'A2.3').
has_ui(finding, 'T033').
has_tree_number(finding, 'A2.2').
has_ui(laboratory_or_test_result, 'T034').
has_tree_number(laboratory_or_test_result, 'A2.2.1').
has_ui(injury_or_poisoning, 'T037').
has_tree_number(injury_or_poisoning, 'B2.3').
has_ui(biologic_function, 'T038').
has_tree_number(biologic_function, 'B2.2.1').
has_ui(physiologic_function, 'T039').
has_tree_number(physiologic_function, 'B2.2.1.1').
has_ui(organism_function, 'T040').
has_tree_number(organism_function, 'B2.2.1.1.1').
has_ui(mental_process, 'T041').
has_tree_number(mental_process, 'B2.2.1.1.1.1').
has_ui(organ_or_tissue_function, 'T042').
has_tree_number(organ_or_tissue_function, 'B2.2.1.1.2').
has_ui(cell_function, 'T043').
has_tree_number(cell_function, 'B2.2.1.1.3').
has_ui(molecular_function, 'T044').
has_tree_number(molecular_function, 'B2.2.1.1.4').
has_ui(genetic_function, 'T045').
has_tree_number(genetic_function, 'B2.2.1.1.4.1').
has_ui(pathologic_function, 'T046').
has_tree_number(pathologic_function, 'B2.2.1.2').
has_ui(disease_or_syndrome, 'T047').
has_tree_number(disease_or_syndrome, 'B2.2.1.2.1').
has_ui(mental_or_behavioral_dysfunction, 'T048').
has_tree_number(mental_or_behavioral_dysfunction, 'B2.2.1.2.1.1').
has_ui(cell_or_molecular_dysfunction, 'T049').
has_tree_number(cell_or_molecular_dysfunction, 'B2.2.1.2.2').
has_ui(experimental_model_of_disease, 'T050').
has_tree_number(experimental_model_of_disease, 'B2.2.1.2.3').
has_ui(event, 'T051').
has_tree_number(event, 'B').
has_ui(activity, 'T052').
has_tree_number(activity, 'B1').
has_ui(behavior, 'T053').
has_tree_number(behavior, 'B1.1').
has_ui(social_behavior, 'T054').
has_tree_number(social_behavior, 'B1.1.1').
has_ui(individual_behavior, 'T055').
has_tree_number(individual_behavior, 'B1.1.2').
has_ui(daily_or_recreational_activity, 'T056').
has_tree_number(daily_or_recreational_activity, 'B1.2').
has_ui(occupational_activity, 'T057').
has_tree_number(occupational_activity, 'B1.3').
has_ui(health_care_activity, 'T058').
has_tree_number(health_care_activity, 'B1.3.1').
has_ui(laboratory_procedure, 'T059').
has_tree_number(laboratory_procedure, 'B1.3.1.1').
has_ui(diagnostic_procedure, 'T060').
has_tree_number(diagnostic_procedure, 'B1.3.1.2').
has_ui(therapeutic_or_preventive_procedure, 'T061').
has_tree_number(therapeutic_or_preventive_procedure, 'B1.3.1.3').
has_ui(research_activity, 'T062').
has_tree_number(research_activity, 'B1.3.2').
has_ui(molecular_biology_research_technique, 'T063').
has_tree_number(molecular_biology_research_technique, 'B1.3.2.1').
has_ui(governmental_or_regulatory_activity, 'T064').
has_tree_number(governmental_or_regulatory_activity, 'B1.3.3').
has_ui(educational_activity, 'T065').
has_tree_number(educational_activity, 'B1.3.4').
has_ui(machine_activity, 'T066').
has_tree_number(machine_activity, 'B1.4').
has_ui(phenomenon_or_process, 'T067').
has_tree_number(phenomenon_or_process, 'B2').
has_ui(human_caused_phenomenon_or_process, 'T068').
has_tree_number(human_caused_phenomenon_or_process, 'B2.1').
has_ui(environmental_effect_of_humans, 'T069').
has_tree_number(environmental_effect_of_humans, 'B2.1.1').
has_ui(natural_phenomenon_or_process, 'T070').
has_tree_number(natural_phenomenon_or_process, 'B2.2').
has_ui(entity, 'T071').
has_tree_number(entity, 'A').
has_ui(physical_object, 'T072').
has_tree_number(physical_object, 'A1').
has_ui(manufactured_object, 'T073').
has_tree_number(manufactured_object, 'A1.3').
has_ui(medical_device, 'T074').
has_tree_number(medical_device, 'A1.3.1').
has_ui(research_device, 'T075').
has_tree_number(research_device, 'A1.3.2').
has_ui(conceptual_entity, 'T077').
has_tree_number(conceptual_entity, 'A2').
has_ui(idea_or_concept, 'T078').
has_tree_number(idea_or_concept, 'A2.1').
has_ui(temporal_concept, 'T079').
has_tree_number(temporal_concept, 'A2.1.1').
has_ui(qualitative_concept, 'T080').
has_tree_number(qualitative_concept, 'A2.1.2').
has_ui(quantitative_concept, 'T081').
has_tree_number(quantitative_concept, 'A2.1.3').
has_ui(spatial_concept, 'T082').
has_tree_number(spatial_concept, 'A2.1.5').
has_ui(geographic_area, 'T083').
has_tree_number(geographic_area, 'A2.1.5.4').
has_ui(molecular_sequence, 'T085').
has_tree_number(molecular_sequence, 'A2.1.5.3').
has_ui(nucleotide_sequence, 'T086').
has_tree_number(nucleotide_sequence, 'A2.1.5.3.1').
has_ui(amino_acid_sequence, 'T087').
has_tree_number(amino_acid_sequence, 'A2.1.5.3.2').
has_ui(carbohydrate_sequence, 'T088').
has_tree_number(carbohydrate_sequence, 'A2.1.5.3.3').
has_ui(regulation_or_law, 'T089').
has_tree_number(regulation_or_law, 'A2.4.2').
has_ui(occupation_or_discipline, 'T090').
has_tree_number(occupation_or_discipline, 'A2.6').
has_ui(biomedical_occupation_or_discipline, 'T091').
has_tree_number(biomedical_occupation_or_discipline, 'A2.6.1').
has_ui(organization, 'T092').
has_tree_number(organization, 'A2.7').
has_ui(health_care_related_organization, 'T093').
has_tree_number(health_care_related_organization, 'A2.7.1').
has_ui(professional_society, 'T094').
has_tree_number(professional_society, 'A2.7.2').
has_ui(self_help_or_relief_organization, 'T095').
has_tree_number(self_help_or_relief_organization, 'A2.7.3').
has_ui(group, 'T096').
has_tree_number(group, 'A2.9').
has_ui(professional_or_occupational_group, 'T097').
has_tree_number(professional_or_occupational_group, 'A2.9.1').
has_ui(population_group, 'T098').
has_tree_number(population_group, 'A2.9.2').
has_ui(family_group, 'T099').
has_tree_number(family_group, 'A2.9.3').
has_ui(age_group, 'T100').
has_tree_number(age_group, 'A2.9.4').
has_ui(patient_or_disabled_group, 'T101').
has_tree_number(patient_or_disabled_group, 'A2.9.5').
has_ui(group_attribute, 'T102').
has_tree_number(group_attribute, 'A2.8').
has_ui(chemical, 'T103').
has_tree_number(chemical, 'A1.4.1').
has_ui(chemical_viewed_structurally, 'T104').
has_tree_number(chemical_viewed_structurally, 'A1.4.1.2').
has_ui(organic_chemical, 'T109').
has_tree_number(organic_chemical, 'A1.4.1.2.1').
has_ui(nucleic_acid_nucleoside_or_nucleotide, 'T114').
has_tree_number(nucleic_acid_nucleoside_or_nucleotide, 'A1.4.1.2.1.5').
has_ui(amino_acid_peptide_or_protein, 'T116').
has_tree_number(amino_acid_peptide_or_protein, 'A1.4.1.2.1.7').
has_ui(chemical_viewed_functionally, 'T120').
has_tree_number(chemical_viewed_functionally, 'A1.4.1.1').
has_ui(pharmacologic_substance, 'T121').
has_tree_number(pharmacologic_substance, 'A1.4.1.1.1').
has_ui(biomedical_or_dental_material, 'T122').
has_tree_number(biomedical_or_dental_material, 'A1.4.1.1.2').
has_ui(biologically_active_substance, 'T123').
has_tree_number(biologically_active_substance, 'A1.4.1.1.3').
has_ui(hormone, 'T125').
has_tree_number(hormone, 'A1.4.1.1.3.2').
has_ui(enzyme, 'T126').
has_tree_number(enzyme, 'A1.4.1.1.3.3').
has_ui(vitamin, 'T127').
has_tree_number(vitamin, 'A1.4.1.1.3.4').
has_ui(immunologic_factor, 'T129').
has_tree_number(immunologic_factor, 'A1.4.1.1.3.5').
has_ui(indicator_reagent_or_diagnostic_aid, 'T130').
has_tree_number(indicator_reagent_or_diagnostic_aid, 'A1.4.1.1.4').
has_ui(hazardous_or_poisonous_substance, 'T131').
has_tree_number(hazardous_or_poisonous_substance, 'A1.4.1.1.5').
has_ui(substance, 'T167').
has_tree_number(substance, 'A1.4').
has_ui(food, 'T168').
has_tree_number(food, 'A1.4.3').
has_ui(functional_concept, 'T169').
has_tree_number(functional_concept, 'A2.1.4').
has_ui(intellectual_product, 'T170').
has_tree_number(intellectual_product, 'A2.4').
has_ui(language, 'T171').
has_tree_number(language, 'A2.5').
has_ui(sign_or_symptom, 'T184').
has_tree_number(sign_or_symptom, 'A2.2.2').
has_ui(classification, 'T185').
has_tree_number(classification, 'A2.4.1').
has_ui(anatomical_abnormality, 'T190').
has_tree_number(anatomical_abnormality, 'A1.2.2').
has_ui(neoplastic_process, 'T191').
has_tree_number(neoplastic_process, 'B2.2.1.2.1.2').
has_ui(receptor, 'T192').
has_tree_number(receptor, 'A1.4.1.1.3.6').
has_ui(archaeon, 'T194').
has_tree_number(archaeon, 'A1.1.1').
has_ui(antibiotic, 'T195').
has_tree_number(antibiotic, 'A1.4.1.1.1.1').
has_ui(element_ion_or_isotope, 'T196').
has_tree_number(element_ion_or_isotope, 'A1.4.1.2.3').
has_ui(inorganic_chemical, 'T197').
has_tree_number(inorganic_chemical, 'A1.4.1.2.2').
has_ui(clinical_drug, 'T200').
has_tree_number(clinical_drug, 'A1.3.3').
has_ui(clinical_attribute, 'T201').
has_tree_number(clinical_attribute, 'A2.3.1').
has_ui(drug_delivery_device, 'T203').
has_tree_number(drug_delivery_device, 'A1.3.1.1').
has_ui(eukaryote, 'T204').
has_tree_number(eukaryote, 'A1.1.3').

% UMLS Ontology Rules
% Useful rules for querying the ontology

% Transitive closure of is-a relationships
% subclass(X, Y) - X is a subclass of Y (direct or indirect)
subclass(X, Y) :- isa(X, Y).
subclass(X, Z) :- isa(X, Y), subclass(Y, Z).

% Sibling relationships
% sibling(X, Y) - X and Y share a common parent
sibling(X, Y) :- isa(X, Z), isa(Y, Z), X \= Y.

% Ancestor relationships
% ancestor(X, Y) - X is an ancestor of Y
ancestor(X, Y) :- isa(Y, X).
ancestor(X, Z) :- isa(Y, X), ancestor(Y, Z).

% Descendant relationships
% descendant(X, Y) - X is a descendant of Y
descendant(X, Y) :- ancestor(Y, X).

% Common ancestor
% common_ancestor(X, Y, Z) - Z is a common ancestor of X and Y
common_ancestor(X, Y, Z) :- ancestor(Z, X), ancestor(Z, Y), X \= Y.

% Related concepts
% related(X, Y, R) - X is related to Y by relation R
related(X, Y, R) :- semantic_relation(X, R, Y).
related(X, Y, R) :- semantic_relation(Y, R, X).

% Concepts with definitions
% has_definition(X) - X has a definition
has_definition(X) :- definition(X, _).

% Example Queries
% Use these queries to explore the UMLS ontology

% Find all concepts
% ?- concept(X, _, _).

% Find all subclasses of a concept
% ?- subclass(X, organism).

% Find all ancestors of a concept
% ?- ancestor(X, plant).

% Find siblings of a concept
% ?- sibling(plant, X).

% Find concepts with definitions
% ?- has_definition(X), definition(X, Def).

% Find semantic relations involving a concept
% ?- related(organism, Y, R).

% Find common ancestors
% ?- common_ancestor(plant, animal, X).


% Generation Statistics
% Concepts: 182
% Hierarchical relations: 181
% Semantic relations: 0
% Definitions: 127
% Properties: 254

% End of UMLS Ontology