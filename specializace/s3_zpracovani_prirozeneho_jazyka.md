# S3. Zpracování přirozeného jazyka

> Korpusy, jazykové modely. Automatické morfologické a syntaktické značkování. Klasifikace textů, extrakce informací. Rekurentní neuronové sítě pro jazykové modelování, zpracování sekvencí, transformery. Odpovídání na otázky, strojový překlad. (PA153)

https://hackmd.io/@uizd-suui/HkRhtUPfL

# Natural Language Processing (NLP) Learning Document

## Key Topics

### Corpora and Language Models

#### Corpora
- **Definition**: A corpus is a collection of texts in natural language used for linguistic analysis.
- **Types**:
  - **Monolingual** and **Multilingual**
  - **Synchronous** and **Diachronic** (studying language trends over time)
  - **Parallel** and **Comparable Parallel** (used for translation tasks)
  - **Full Texts** or **Samples**
  - **Text-Based** or **Multimodal** (including audio/video data)
- **Examples**:
  - **Brown Corpus**
  - **British National Corpus (BNC)**
  - **Corpus of Contemporary American English (COCA)**
  - **Czech National Corpus (CNC)**
- **Challenges**:
  - Copyright issues
  - High costs
  - Data acquisition difficulties
  - Managing noise and redundancy

#### Language Models
- **Definition**: A language model is a computational model designed to predict the likelihood of word sequences. It is widely used in natural language processing tasks such as text generation, speech recognition, and machine translation.
- **Purpose**: Predict the likelihood of sequences of words (e.g., for text generation, translation).
- **Types**:
  - **Statistical Language Models**: Based on word probabilities and n-grams.
  - **Neural Language Models**: Employ deep learning, such as recurrent neural networks (RNNs) and transformers.

### Morphological and Syntactic Tagging

#### Morphological Analysis
- **Definition**: Morphological analysis is the process of analyzing the structure of words to identify their root form (lemma) and grammatical features. It involves segmenting a word into morphemes, which are the smallest units of meaning or grammatical function in a language.
- **Goal**: Analyze words to retrieve their lemma and grammatical features (e.g., case, number, gender).
- **Challenges**:
  - Handling exceptions (e.g., negation, irregular forms).
  - Ambiguities in morphological analysis.
- **Tagging Systems**:
  - **Positional Tagging**: Fixed positions represent grammatical categories.
  - **Attribute-Value Pair Systems**: Flexible, readable by regex, and expandable.

#### Syntactic Tagging
- **Definition**: Syntactic tagging, or parsing, is the process of analyzing the syntactic structure of sentences by identifying relationships between words and building hierarchical structures (e.g., parse trees).
- **Purpose**: Determine sentence structure and relationships between words.
- **Approaches**:
  - Dependency Parsing: Maps syntactic relationships as dependencies between words.
  - Constituency Parsing: Builds a tree structure based on phrase hierarchies.
- **Applications**:
  - Grammar checks
  - Information extraction
  - Question answering

### Text Classification and Information Extraction

#### Text Classification
- **Definition**: Text classification is the process of categorizing or labeling text data into predefined classes based on its content.
- **Techniques**:
  - Rule-Based Approaches
  - Machine Learning Models (e.g., Naïve Bayes, SVMs)
  - Neural Models (e.g., transformers, RNNs)

#### Information Extraction
- **Definition**: Information extraction refers to the process of automatically extracting structured and relevant information from unstructured text data.
- **Goal**: Extract structured information from unstructured text.
- **Types**:
  - Named Entity Recognition (NER): Identifies entities like names, locations, dates.
  - Relation Extraction: Finds relationships between entities.
  - Event Extraction: Identifies and classifies events in text.

### Neural Networks for Language Modeling

#### Recurrent Neural Networks (RNNs)
- **Definition**: Recurrent Neural Networks are a class of neural networks designed for sequential data, where connections between nodes form a directed graph along a sequence.
- **Strengths**: Process sequential data; can model long-term dependencies.
- **Limitations**: Vanishing gradient problem in long sequences.

#### Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRUs)
- **Definition**: LSTMs and GRUs are specialized types of RNNs designed to capture long-term dependencies and address the vanishing gradient problem.
- **Improvements**: Address vanishing gradient issues; effective for longer sequences.

#### Transformers
- **Definition**: Transformers are neural network architectures that use self-attention mechanisms to process sequences of data in parallel, significantly improving performance on NLP tasks.
- **Core Idea**: Use self-attention mechanisms to process all words in a sequence simultaneously.
- **Key Models**:
  - BERT (Bidirectional Encoder Representations from Transformers): Pretrained for understanding context.
  - GPT (Generative Pretrained Transformer): Focuses on text generation.

### Question Answering and Machine Translation

#### Question Answering (QA)
- **Definition**: Question answering is a field of NLP focused on building systems that automatically answer questions posed in natural language by retrieving or generating appropriate responses.
- **Types**:
  - Extractive QA: Extracts answers from a provided context.
  - Generative QA: Generates answers from scratch.

#### Machine Translation (MT)
- **Definition**: Machine translation involves the automatic conversion of text from one language to another while preserving meaning and fluency.
- **Types**:
  - Rule-Based MT (RBMT): Uses linguistic rules for translation.
  - Statistical MT (SMT): Relies on large bilingual corpora to learn translation patterns.
  - Neural MT (NMT): Leverages deep learning, primarily with transformers.
- **Key Concepts**:
  - Vauquois Triangle: Describes levels of analysis for translation (e.g., direct translation, interlingua).

## Resources
- **Corpora Tools**: Word Sketch Engine, Bonito (with Corpus Query Language).
- **NLP Tools**:
  - Morphological Analyzers: Majka, Ajka
  - Parsers: Synt, VaDis
  - Annotators: DESAMB, WordNet
- **Useful Links**:
  - [Prague Dependency Treebank](https://ufal.mff.cuni.cz/pdt)
  - [WordNet](https://wordnet.princeton.edu)

## Summary
This document provides an overview of essential NLP topics, from corpora and language models to advanced neural architectures like transformers. It also covers practical applications like text classification, question answering, and machine translation, offering foundational knowledge for deeper exploration.
