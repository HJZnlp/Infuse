from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import torch
import numpy as np
from itertools import zip_longest
import argparse
from tqdm import tqdm
import stanza

def get_sents(c, seg_tokenizer):
    doc =seg_tokenizer(c)
    sents=[sentence.text for sentence in doc.sentences ]
    return sents


# segment doc and summ into nested
def seg_raw_data(docs, sums):
    seg_tokenizer = stanza.Pipeline(lang="en", processors='tokenize')
    if len(docs)!=len(sums):
            print("unmatch no of samples.")
            print(f"doc:{len(docs)} sum: {len(sums)}")
            exit()
    print(f"load {len(docs)} pairs")

    doc_sents,summ_sents=[],[]

    for doc, summ in zip(docs,sums):
        doc_sents.append(get_sents(doc,seg_tokenizer))
        summ_sents.append(get_sents(summ,seg_tokenizer))

    return doc_sents,summ_sents


def load_data(doc_name, sum_name):
    # Read documents; split into lists by empty lines.
    with open(doc_name, 'r') as fdoc:
        doc_content = fdoc.read().strip()
        docs = [doc.split('\n') for doc in doc_content.split('\n\n')]

    with open(sum_name, 'r') as fsum:
        sum_content = fsum.read().strip()
        sums = [summary.split('\n') for summary in sum_content.split('\n\n')]

    if len(docs)!=len(sums):
            print("unmatch no of samples.")
            print(f"doc:{len(docs)} sum: {len(sums)}")
            exit()
    print(f"load {len(docs)} pairs")

    return docs, sums


def grouped(iterable, n):
    return zip_longest(*[iter(iterable)] * n)

def generate_ranges(max_number, bz=16):
    start = 1
    end = bz

    while start <= max_number:
        yield range(start, min(end, max_number) + 1)
        start += bz
        end += bz



class INFUSE:
    def __init__(self, model_name, device='cuda:0'):
        self.entailment_idx, self.neutral_idx, self.contradiction_idx = 0, 2, 1
        self.max_input_length = 512
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.model.to(device)
        self.model.half()  # use fp16 as in summac

    def run_model_batch(self, sequence_raw, hypothesis_raw):
        batch_tokens = self.tokenizer.batch_encode_plus(list(zip(sequence_raw, hypothesis_raw)), padding=True, truncation=True, max_length=self.max_input_length, return_tensors="pt" ,truncation_strategy="only_first")
        with torch.no_grad():
            model_outputs = self.model(**{k: v.to(self.device) for k, v in batch_tokens.items()})
        batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
        batch_evids = batch_probs[:, self.entailment_idx].tolist()
        batch_conts = batch_probs[:, self.contradiction_idx].tolist()
        batch_neuts = batch_probs[:, self.neutral_idx].tolist()
        return batch_evids, batch_neuts, batch_conts

    #  go through all sent to obtain d2s and s2d matrics
    def obtain_matrices(self,doc_sentences, sentence, bz=24):
        doc2sum_res, sum2doc_res = [], []

        for doc_batch in grouped(doc_sentences, bz):
            cleaned_batch = list(filter(None, doc_batch))
            hypotheses = [sentence] * len(cleaned_batch)

            d2s_ent, _, _ = self.run_model_batch(cleaned_batch, hypotheses)
            s2d_ent, _, _ = self.run_model_batch(hypotheses, cleaned_batch)

            doc2sum_res.extend(d2s_ent)
            sum2doc_res.extend(s2d_ent)

        return doc2sum_res, sum2doc_res

    def create_candidate_dict(self,doc2sum_scores, sum2doc_scores, doc_length, super_K):
        if 2 * super_K > doc_length:
            idx_doc2sum = np.argsort(doc2sum_scores)
        else:
            idx_doc2sum = np.argsort(doc2sum_scores)[-super_K * 2:]

        return {idx: [doc2sum_scores[idx], sum2doc_scores[idx]] for idx in idx_doc2sum}


    def build_premise(self,candidate_dict, doc_sentences, super_K, rev):
        sorted_candidates = sorted(
            candidate_dict.items(),
            key=lambda item: (item[1][0] + item[1][1], item[1][0]) if rev else item[1][0],
            reverse=True
        )
        return "".join(doc_sentences[idx] for idx, _ in sorted(sorted_candidates[:super_K], key=lambda item: item[0]))


    def evaluate_batch_premises(self,premise_list, sentence):
        hypotheses = [sentence] * len(premise_list)
        ent_scores, neu_scores, _ = self.run_model_batch(premise_list, hypotheses)
        return ent_scores, neu_scores


    def update_scores(self,batch_ent, batch_neu, best_neu_score, best_ent_score, sentence_score):
        for i, score_neu in enumerate(batch_neu):
            if score_neu > best_neu_score:
                sentence_score=best_ent_score
                break
            else:
                best_neu_score = score_neu
                best_ent_score = batch_ent[i]


    def process_document_summary(self, doc_path, summ_path, rev, seg):
        if seg:
            doc,summ=seg_raw_data(doc_path,summ_path)
        else:
            doc,summ=load_data(doc_path,summ_path)

        scorer = []

        for ind, summary_sentences in tqdm(enumerate(summ),total=len(summ) ):
            sample_score=[]
            doc_sentences = doc[ind]
            doc_length = len(doc_sentences)

            for summary_sentence in summary_sentences:
                sentence_score = -1
                best_ent_score = -1
                best_neu_score = 1000

                doc2sum_res, sum2doc_res = self.obtain_matrices(doc_sentences, summary_sentence)


                for super_K_range in generate_ranges(len(doc_sentences)):
                    premise_list=[]
                    if sentence_score >0:
                        break
                    for super_K in super_K_range:
                        candidate_dict = self.create_candidate_dict(doc2sum_res, sum2doc_res, doc_length, super_K )
                        cur_premise = self.build_premise(candidate_dict, doc_sentences, super_K,rev)
                        premise_list.append(cur_premise)

                    batch_ent, batch_neu = self.evaluate_batch_premises(premise_list, summary_sentence)


                    for i, score_neu in enumerate(batch_neu):
                        if score_neu > best_neu_score:
                            sentence_score=best_ent_score
                            break
                        else:
                            best_neu_score = score_neu
                            best_ent_score = batch_ent[i]

                if sentence_score ==-1:
                    sentence_score=best_ent_score
                sample_score.append(sentence_score)
            scorer.append(sample_score)

        return scorer


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_doc", type=str, default='test')
    parser.add_argument("--input_sum", type=str, default='test')
    parser.add_argument("--save_address", type=str, default='test')
    parser.add_argument("--reverse", type=int,default=1)
    parser.add_argument("--seg", type=int,default=0)
    parser.add_argument("--model_name", type=str,default="tals/albert-xlarge-vitaminc-mnli")
    args = parser.parse_args()
    doc_path=args.input_doc.strip()
    sum_path=args.input_sum.strip()
    out_path=args.save_address.strip()
    rev=args.reverse.strip()
    seg=args.seg.strip()
    model_name=args.model_name.strip()

    model=INFUSE(model_name)

    scorer=model.process_document_summary(doc_path,sum_path,rev,seg)


    print("done")

    with open(out_path, "w") as fout:
        for summ_sample in scorer:
            # Join all numeric scores into a string separated by spaces for each summary sample.
            out_line = ' '.join(map(str, summ_sample)) + '\n'
            fout.write(out_line)
