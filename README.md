# MIA 2020 Shared Task on Cross-lingual Open-Retrieval Question Answering. 

### Quick Links

- [Datasets](#datasets)
- [Download](#download-scripts)
- [Format](#dataset-format)
- [Evaluate](#evaluation)
- [Baseline](#baseline-model)
- [Submission](#submission)

## Datasets

We have adapted several existing datasets from their original formats and settings to conform to our unified extractive setting. Most notably:

- We provide only a single, length-limited context.
- There are no unanswerable or non-span answer questions.
- All questions have at least one accepted answer that is found exactly in the context.

A span is judged to be an exact match if it matches the answer string after performing normalization consistent with the [SQuAD](https://stanford-qa.com) dataset. Specifically:

- The text is uncased.
- All punctuation is stripped.
- All articles `{a, an, the}` are removed.
- All consecutive whitespace markers are compressed to just a single normal space `' '`.

### Training Data

| Dataset | Download | # of Examples |
| :-----: | :-------:| :------: |
| [Natural Questions]() | [Link]() |  |
| [XOR-TyDi QA]() | [Link]() |  |


### Development Data

| Dataset | Download |  # of Examples |
| :-----: | :-------:| :------: |
| [XOR-TyDi QA]() | [Link]() |  |
| [MKQA]() | [Link]() |  |


## Evaluations


## Baseline Models

## Submissions
