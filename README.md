# INFUSE

This repository contains the code and data for EACL2024 paper: Fine-Grained Natural Language Inference Based Faithfulness Evaluation for Diverse Summarisation Tasks

INFUSE is a faithfulness evaluation approach that **IN**crementally reasons over a document so as to arrive at a **F**aithf**u**lnes**s** **E**stimation of its summary. This repository contains the implementation of INFUSE, as well as **Diversumm**, a faithfulness evaluation benchmark on long document summarisation with diverse domains and genres and multi-document summarisation.

Should you have any queries please contact me at v1hzha17@ed.ac.uk

## Quickstart

```bash
git clone https://github.com/HJZnlp/Infuse.git
cd Infuse
pip install -r requirements.txt
```

## Example Use
### Direct use

```
from src.infuse import INFUSE

documents=["document_a","document_b"......]
summaries=["summary_a","summary_b"......]
require_segmentation=1

model=INFUSE(YOUR_NLI_MODEL_NAME)

scorer=model.process_document_summary(documents,summaries,require_segmentation)
# scorer will return a nest list of scores for each summary sentence

```

### Bash
```
doc_path = YOUR_DOCUMENT_PATH
sum_path = YOUR_SUMMARY_PATH
outpath = YOUR_OUTPUT_PATH
python src/infuse.py --input_doc $doc_path --input_sum $sum_path --save_address $outpath
```

Ensure that the document and summary are preprocessed to meet the following format criteria before running the script:

1. Segment both the document and summary into individual sentences.
2. Separate each sentence with a newline character ("\n").
3. Separate each example (consisting of pairs or groups of sentences) with two newline characters ("\n\n").

Note: Replace YOUR_DOCUMENT_PATH, YOUR_SUMMARY_PATH, and YOUR_OUTPUT_PATH with the actual file paths on your system.

## Citation
```
@inproceedings{zhang-etal-2024-fine,
    title = "Fine-Grained Natural Language Inference Based Faithfulness Evaluation for Diverse Summarisation Tasks",
    author = "Zhang, Huajian  and
      Xu, Yumo  and
      Perez-Beltrachini, Laura",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.102",
    pages = "1701--1722",
}
```
