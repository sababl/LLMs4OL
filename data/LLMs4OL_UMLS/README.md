# UMLS Dataset for LLMs4OL

This dataset contains UMLS (Unified Medical Language System) data formatted for use with the LLMs4OL framework.

## Dataset Statistics

- **Semantic Types**: 127
- **Total Terms**: 297
- **Entity Typing Samples**: 297
- **Hierarchy Levels**: 6
- **Label Mapper Entries**: 127

## Files

- `label_mapper.json`: Maps UMLS tree numbers/semantic types to various label representations
- `hierarchy.json`: Hierarchical structure of UMLS semantic types
- `templates.json`: Template questions for entity typing tasks
- `processed/entity_typing.json`: Entity typing dataset for TaskA
- `umls_stats.json`: Dataset statistics

## Usage

This dataset is designed for TaskA (Entity Typing) in the LLMs4OL framework. It can be used to evaluate how well language models can classify medical terms according to UMLS semantic types.

## Domain

Medical and biomedical terminology from the Unified Medical Language System (UMLS).

## Format

The dataset follows the LLMs4OL standard format for entity typing tasks.
