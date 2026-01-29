# encoding=utf8
from common import *

# from prover.proof import ProofStep, Proof, InvalidProofStep
# from prover.search import ProofGraph
from proof import ProofStep, Proof, InvalidProofStep
from search import ProofGraph
import numpy as np
import os
import json
import torch
import itertools
import pytorch_lightning as pl
import sys
from mytransformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    LogitsProcessor,
)
# from prover.evaluate import evaluate_entailmentbank, evaluate_ruletaker
from evaluate import evaluate_entailmentbank, evaluate_ruletaker

from task1_policy import Policy
from task1_parse import parse_args
from utils import get_intermediate_sentence, ParaPattern, jaccard_similarity, \
    aggregate_ancestor, normalize, to_binary_path
import re

import sys
from verifier.model import EntailmentClassifier

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from bleurt import score
from StepScorer import StepScorer

from cosine_similarity import CosineSimilarity
import warnings
warnings.filterwarnings("ignore")


class EntailmentWriter(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        stepwise: bool,
        max_num_steps: int,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_input_len: int,
        proof_search: bool,
        verifier_weight: float,
        verifier_ckpt: Optional[str] = None,
        oracle_prover: Optional[bool] = False,
        oracle_verifier: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.stepwise = stepwise
        self.max_num_steps = max_num_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk
        self.verifier_weight = verifier_weight
        self.proof_search = proof_search
        self.oracle_prover = oracle_prover
        self.oracle_verifier = oracle_verifier
        if stepwise and verifier_weight > 0:
            assert verifier_weight <= 1.0
            assert verifier_ckpt is not None
            self.verifiers = [
                EntailmentClassifier.load_from_checkpoint(verifier_ckpt)
            ]  # Avoid making the verifier a submodule.

        model_name_base = 'path/T5-large'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_base, model_max_length=max_input_len
        )
        self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name_base)
        self.discount = 0.9
        self.wrong_node_reward = -1 

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        sentence_span_list,
        sentence_current_label,
        golden_proof
    ) -> Any:

        model_output = self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = model_output.loss

        lm_loss_step = []
        encoder_hidden = []
        reward = []
        rl_loss = []
        for i, input_id in enumerate(input_ids):
            input_id = input_id[input_id != 0]
            input_id_list = input_id.tolist()            
            input_text = self.tokenizer.decode(input_id_list)
            output_text, output_scores = self.generate_proof_step_RL(input_text)

            conclusion_sents = {}
            hypothesis = ''
            match_hy = re.search(r'\$hypothesis\$ = (.*?)(?= ;|\Z)', input_text)
            if match_hy:
                hypothesis = match_hy.group(1)

            try:
                choice = []
                pattern = r'\$proof\$ = (.*?)</s>'
                match = re.search(pattern, input_text)
                if match:
                    extracted_proof = match.group(1)            
                    for s in extracted_proof.split(";"):
                        s = s.strip()
                        if s == "":
                            continue
                        if s.endswith(";"):
                            s = s[:-1]
                        if s.count(" -> ") != 1:
                            raise InvalidProofStep
                        ex_premises, ex_conclusion = s.split(" -> ")

                        ex_premise_idents = []
                        for p in ex_premises.split(" & "):
                            if not re.fullmatch(r"(sent|int)\d+", p):
                                raise InvalidProofStep
                            ex_premise_idents.append(p)
                        choice.append(ex_premise_idents)
                        m = re.fullmatch(r"(?P<ident>int\d+): (?P<sent>.+)", ex_conclusion)
                        conclusion_sents[' '.join(sorted(ex_premise_idents))] = m["sent"]
                
                premise_idents = []
                step_present = output_text[0]
                if step_present.endswith(";"):
                    step_present = step_present[:-1]
                if step_present.count(" -> ") != 1:
                    raise InvalidProofStep
                premises, conclusion = step_present.split(" -> ")
                for p in premises.split(" & "):
                    if not re.fullmatch(r"(sent|int)\d+", p):
                        raise InvalidProofStep
                    premise_idents.append(p)
                if conclusion == "hypothesis":
                    conclusion_sents[' '.join(premise_idents)] = hypothesis
                else:
                    m = re.fullmatch(r"(?P<ident>int+): (?P<sent>.+)", conclusion)
                    conclusion_sents[' '.join(sorted(premise_idents))] = m["sent"]
                choice.append(premise_idents) 

                golden_step = []
                for step in golden_proof[i].proof_steps:
                    golden_step.append(step.premise_idents)
                step_source = np.array(golden_step)

                try:
                    cur_reward = self.get_stepwise_reward_align_accu_log(step_source, choice, self.discount, hypothesis, conclusion_sents)
                    # cur_reward = self.get_stepwise_reward_align_accu_log(step_source, choice, self.discount)
                except:
                    cur_reward = [0.0]
                cur_reward_new = torch.tensor(cur_reward)

                logits = model_output.logits[i].unsqueeze(0)

                factors = torch.ones_like(logits)
                label = labels[i]

                logits_re = F.log_softmax(logits, dim=-1)
                logits_prob = logits_re
                label_new = label[label != -100]
                RL_one = torch.tensor(0.0).to(logits.device)
                baseline = 0.0
                if cur_reward_new.size(0) > 1:
                    baseline = cur_reward_new.mean().item()
                if cur_reward_new.size(0) > 0 and cur_reward_new[-1] > 0:
                    for num, j in enumerate(label_new):
                        RL_one_new = - logits_prob[0, num, j] * (cur_reward_new[-1] - baseline)
                        RL_one = RL_one + RL_one_new

                RL_one = RL_one / (label_new.size(0) + 1e-8)
                rl_loss.append(RL_one)   
            except:
                pass

        loss_RL = torch.stack(rl_loss).sum() / input_ids.size(0)
        loss_total = lm_loss + loss_RL

        with open("path_loss.txt", "a", encoding="utf-8") as f:
            f.write(f"step: {self.num_step}, lm: {lm_loss}, rl: {loss_RL}\n")

        return lm_loss + loss_RL

    def get_stepwise_reward_align_accu_log(self, gold, pred, discount, hypothesis, conclusion_sent):
    # def get_stepwise_reward_align_accu_log(self, gold, pred, discount):

        aggre_leaves = aggregate_ancestor(gold)
        pred_leaves = {}
        pred_map_gold_inter = {}
        each_step_reward = []
        allcorrect = True

        gold_list = [sorted(list(item)) for item in gold.tolist()]
        sorted_pred = [sorted(item) for item in pred]
        pred_dict = {tuple(item): item for item in sorted_pred}
        sorted_pred = [item for item in gold_list if tuple(item) in pred_dict]
        point_align = [False] * len(aggre_leaves)

        total = 0
        for i, pred_step in enumerate(sorted_pred):
            leaves = []
            aligned_pred_step = []
            for name in pred_step:
                if name[0] == 'i':
                    leaves.extend(pred_leaves[name])
                    aligned_pred_step.append(pred_map_gold_inter[name])
                else:
                    leaves.append(name)
                    aligned_pred_step.append(name)

            leaves = list(set(leaves))
            pred_leaves['int' + str(i + 1)] = leaves

            max_sim = 0
            map_gold = 0
            for j, gold_leaves in enumerate(aggre_leaves):
                jaccard_sim = jaccard_similarity(leaves, gold_leaves)
                if jaccard_sim > max_sim:
                    max_sim = jaccard_sim
                    map_gold = j

            if max_sim > 0:
                pred_map_gold_inter['int' + str(i + 1)] = 'int' + str(map_gold + 1)
            else:
                pred_map_gold_inter['int' + str(i + 1)] = 'int0'
            gold_step = gold[map_gold]

            if set(gold_step) == set(aligned_pred_step):
                # each_step_reward.append(1)

                # cosin_s = CosineSimilarity(' '.join(list(gold_step)), ' '.join(aligned_pred_step)).main()
                # each_step_reward.append(cosin_s)

                if i == len(gold_list) - 1 or i == len(gold_list) - 2:
                    conclusion_s = conclusion_sent[' '.join(sorted(pred_step))]
                    H_bleurt_score = self.bleurt_scorer.score(references = [hypothesis], 
                                                        candidates = [conclusion_s])[0]
                    # rew = 1 + H_bleurt_score
                    rew = H_bleurt_score
                    if rew > 0:
                        step_inputs.append({
                            'pre_sent': pred_step.premise_sents,
                            'con_sent': pred_step.conclusion_sent,
                        })
                        step_scores = self.step_scorer(step_inputs) 
                        rew = rew + step_scores
                        each_step_reward.append(rew)
                    else:
                        each_step_reward.append(1)    
                else:
                    each_step_reward.append(1)    

            else:
                each_step_reward.append(self.wrong_node_reward)  
                allcorrect = False

        acc_stepwise_reward = []
        ac_r = 0
        for i, r in enumerate(each_step_reward):
            coefficient = [discount ** ii for ii in range(i + 1)] 
            acc_stepwise_reward.append(sum(coefficient) * r + ac_r) 

        return acc_stepwise_reward
    
    def move_verifier_to_device(self) -> None:
        if hasattr(self, "verifiers"):
            self.verifiers[0].to(self.device)

    def on_train_start(self) -> None:
        self.move_verifier_to_device()
        self.num_step = 0
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        self.move_verifier_to_device()

    def on_test_start(self) -> None:
        self.move_verifier_to_device()
        self.bleurt_scorer = score.BleurtScorer('path/bleurt-large-512') 
        self.step_scorer = StepScorer('path/step_scorer/z9TPfknY', device='cuda')

    def generate_entire_proof(
        self, input_text: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Single-shot proof generation with text-to-text transformers.
        """
        assert self.trainer is not None
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        scores = output.sequences_scores.detach().exp().tolist()
        return output_text, scores

    def generate_stepwise_proof(
        self, proof_gt: List[Proof], batch_idx: int
    ) -> Tuple[List[str], List[float]]:
        """
        Stepwise proof generation.
        """
        proof_pred, step_scores = self.generate_greedy_proofs(proof_gt)
        if not self.proof_search:
            proof_text_pred = [pt.proof_text for pt in proof_pred]
            score = [min(s) if len(s) > 0 else 0.0 for s in step_scores]
        else:
            batch_size = len(proof_gt)
            proof_text_pred = []
            score = []
            for i in range(batch_size):
                p, s = self.search_proof(proof_gt[i], proof_pred[i], step_scores[i])
                proof_text_pred.append(p)
                score.append(s)
        return proof_text_pred, score

    def generate_proof_step(
        self,
        input_text: List[str],
    ) -> Tuple[List[str], Any]:
        """
        Generate a single proof step with text-to-text transformers.
        """
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )

        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=self.topk,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        batch_size = len(input_text)
        assert len(output_text) % batch_size == 0
        k = len(output_text) // batch_size  # k predicted steps for each example.
        output_text = [output_text[i * k : (i + 1) * k] for i in range(batch_size)]

        output_scores = output.sequences_scores.detach().exp().cpu().numpy()
        assert 0.0 <= output_scores.min() <= output_scores.max() <= 1.0
        output_scores = [output_scores[i * k : (i + 1) * k] for i in range(batch_size)]

        return output_text, output_scores
    
    def generate_proof_step_RL(
        self,
        input_text: List[str],
    ) -> Tuple[List[str], Any]:
        """
        Generate a single proof step with text-to-text transformers.
        """
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )

        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=self.topk,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        output_scores = output.sequences_scores.detach().exp().cpu().numpy()

        return output_text, output_scores

    def generate_greedy_proofs(
        self, proof_gt: List[Proof]
    ) -> Tuple[List[Proof], List[List[float]]]:
        """
        Greedily stepwise proof generation.
        """
        all_proof_pred = [
            Proof(pt.context, pt.hypothesis, proof_text="", strict=True)
            for pt in proof_gt
        ]
        proof_pred = all_proof_pred
        unfinished_indexes = list(range(len(proof_gt)))
        all_step_scores: List[List[float]] = [[] for _ in proof_gt]

        for _ in range(self.max_num_steps):
            if len(unfinished_indexes) == 0:
                # All examples in the batch has been finished.
                break
            input_text = [
                f"$hypothesis$ = {pt.hypothesis} ; $context$ = {pt.serialize_context()} ; $proof$ = {'' if pt.proof_text == '' else pt.proof_text + ';'}"
                for pt in proof_pred
            ]
            output_text, output_scores = self.generate_proof_step(input_text)

            proof_steps, prover_scores = self.filter_invalid_steps(
                output_text,
                output_scores,
                proof_pred,
                strict=True,
            )

            scores = self.calculate_score(proof_steps, prover_scores, proof_gt)
            max_s = [s.index(max(s)) if len(s) > 0 else 0 for s in scores]
            proof_steps = [
                steps[max_s[i]] if len(steps) > 0 else None for i, steps in enumerate(proof_steps)
            ]
            scores = [s[max_s[i]] if len(s) > 0 else 0.0 for i,s in enumerate(scores)]

            # Execute the predicted proof steps.
            finished_indexes = []
            for i, j in enumerate(unfinished_indexes):
                step = proof_steps[i]
                if step is None:
                    # Try to get some partial credits.
                    step = self.normalize_predicted_step(
                        output_text[i][0], proof_pred[i]
                    )
                    idx = step.find(";")
                    if idx != -1:
                        step = step[:idx]
                    try:
                        step = ProofStep(proof_pred[i], step, strict=False)
                        if step.is_final():
                            finished_indexes.append(i)
                        proof_pred[i].execute(step)
                        all_step_scores[j].append(float(output_scores[i][0]))
                    except InvalidProofStep:
                        finished_indexes.append(i)
                        proof_pred[i].proof_text = "INVALID_PROOF"
                        # proof_pred[i].proof_text = "sent1 -> hypothesis"
                else:
                    if step.is_final():
                        finished_indexes.append(i)
                    proof_pred[i].execute(step)
                    all_step_scores[j].append(scores[i])

            unfinished_indexes = [
                j for i, j in enumerate(unfinished_indexes) if i not in finished_indexes
            ]
            proof_pred = [
                pt for i, pt in enumerate(proof_pred) if i not in finished_indexes
            ]

        assert (
            pt.is_complete() or pt.proof_text == "INVALID_PROOF"
            for pt in all_proof_pred
        )
        return all_proof_pred, all_step_scores

    def generate_oracle_proof_step(
        self,
        input_text: List[str],
        proof_gt: Proof,
    ) -> Tuple[List[str], List[float]]:
        """
        Oracle prover.
        """
        output_text, output_scores = self.generate_proof_step(input_text)

        for i, inp in enumerate(input_text):
            _, partial_proof = inp.split("$proof$ = ")
            partial_proof = Proof(
                proof_gt.context,
                proof_gt.hypothesis,
                partial_proof,
                strict=True,
            )

            # Add all steps in proof_gt that are valid w.r.t. `partial_proof`.
            for step in proof_gt.proof_steps:
                for sent in step.premise_sents:
                    if sent not in partial_proof:
                        break
                else:
                    premise_idents = []
                    for ident, sent in zip(step.premise_idents, step.premise_sents):
                        if re.fullmatch(r"int\d+", ident):
                            ident = re.search(f"(?P<ident>int\d+): {sent}", inp)[
                                "ident"
                            ]
                        premise_idents.append(ident)
                    premises = " & ".join(premise_idents)
                    if step.conclusion_ident == "hypothesis":
                        conclusion = "hypothesis"
                    else:
                        conclusion = f"int: {step.conclusion_sent}"
                    text = f"{premises} -> {conclusion};"
                    output_text[i].append(text)
                    output_scores[i] = np.append(output_scores[i], 1.0)

        return output_text, output_scores

    def calculate_score(
        self,
        proof_steps: List[List[ProofStep]],
        prover_scores: List[List[float]],
        proof_gt: List[Proof],
    ) -> List[List[float]]:
        if self.verifier_weight == 0:
            return prover_scores

        batch_premises = []
        batch_conclusion = []
        batch_proof_gt = []

        for i, steps in enumerate(proof_steps):
            for s in steps:
                batch_premises.append(s.premise_sents)
                batch_conclusion.append(s.conclusion_sent)
                batch_proof_gt.append(proof_gt[i])

        if self.oracle_verifier:
            verifier_scores = self.calculate_oracle_verifier_score(
                batch_premises,
                batch_conclusion,
                batch_proof_gt,
            )
        else:
            verifier_scores = self.verifiers[0].batch_score(
                batch_premises, batch_conclusion
            )
        
        faithful_scores = []
        for i, pred_steps in enumerate(proof_steps):
            bleurt_inputs = []
            step_inputs = []
            gold_proof = proof_gt[i]
            hypothesis = gold_proof.hypothesis
            for pred_step in pred_steps:
                bleurt_inputs.append(pred_step.conclusion_sent)
                step_inputs.append({
                            'pre_sent': pred_step.premise_sents,
                            'con_sent': pred_step.conclusion_sent,
                        })
            # bleurt_scores = self.bleurt_scorer.score(references = [hypothesis]*len(bleurt_inputs), 
            #                                         candidates = bleurt_inputs)
            step_scores = self.step_scorer(step_inputs) 

            # for b_score, s_score in zip(bleurt_scores, step_scores):
                # faithful_scores.append(0.2 * b_score + s_score)
            for s_score in step_scores:
                faithful_scores.append(s_score)

        faithful_scores = np.array(faithful_scores)

        scores = []
        idx = 0
        for ps in prover_scores:
            if len(ps) == 0:
                scores.append([])
            else:
                scores.append(
                    (
                        (1.0 - self.verifier_weight) * np.array(ps)
                        + self.verifier_weight * (verifier_scores[idx : idx + len(ps)] + faithful_scores[idx : idx + len(ps)])
                    ).tolist()
                )
                idx += len(ps)

        return scores

    def calculate_oracle_verifier_score(
        self,
        batch_premises: List[List[str]],
        batch_conclusion: List[str],
        batch_proof_gt: List[Proof],
    ) -> List[float]:
        """
        Oracle verifier.
        """
        verifier_scores = self.verifiers[0].batch_score(
            batch_premises, batch_conclusion
        )
        assert len(batch_premises) == len(batch_conclusion) == len(batch_proof_gt)

        for i, (premises, conclusion, proof_gt) in enumerate(
            zip(
                batch_premises,
                batch_conclusion,
                batch_proof_gt,
            )
        ):
            for step in proof_gt.proof_steps:
                if (
                    sorted(step.premise_sents) == sorted(premises)
                    and step.conclusion_sent == conclusion
                ):
                    verifier_scores[i] = 1.0
                    break

        return verifier_scores

    def search_proof(
        self,
        proof_gt: Proof,
        proof_greedy: Proof,
        step_scores_greedy: List[float],
    ) -> Tuple[str, float]:
        context, hypothesis = proof_gt.context, proof_gt.hypothesis
        pg = ProofGraph(context, hypothesis)
        pg.initialize(proof_greedy.proof_steps, step_scores_greedy)

        explored_proofs: Set[str] = set()
        context_text = proof_gt.serialize_context()

        while True:
            partial_proof = pg.sample_proof_tree(explored_proofs)
            if partial_proof is None:
                break
            explored_proofs.add(partial_proof)

            input_text = [
                f"$hypothesis$ = {hypothesis} ; $context$ = {context_text} ; $proof$ = {partial_proof}"
            ]
            if self.oracle_prover:
                output_text, output_scores = self.generate_oracle_proof_step(
                    input_text, proof_gt
                )
            else:
                output_text, output_scores = self.generate_proof_step(input_text)
            proof_steps, prover_scores = self.filter_invalid_steps(
                output_text,
                output_scores,
                [Proof(context, hypothesis, partial_proof, strict=False)],
                strict=False,
            )
            scores = self.calculate_score(
                proof_steps,
                prover_scores,
                [proof_gt],
            )

            proof_steps = list(itertools.chain.from_iterable(proof_steps))
            scores = list(itertools.chain.from_iterable(scores))

            graph_updated = False
            for ps, s in zip(proof_steps, scores):
                if pg.expand(ps, s):
                    graph_updated = True
            if not graph_updated:
                break

        proof = pg.extract_proof("hypothesis", rename=True)
        return proof, pg.graph.nodes["hypothesis"]["score"]

    def normalize_predicted_step(self, step: str, proof: Proof) -> str:
        if "-> int:" in step:
            step = step.replace("-> int:", f"-> {proof.next_int()}:").strip()
        return step

    def filter_invalid_steps(
        self,
        output_text: List[str],
        output_scores: List[float],
        proofs: List[Proof],
        strict: bool,
    ) -> Tuple[List[List[ProofStep]], List[List[float]]]:
        batch_size = len(proofs)

        all_proof_steps = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            assert len(output_text[i]) == len(output_scores[i])

            for text, score in zip(output_text[i], output_scores[i]):
                idx = text.find(";")
                if idx != -1:
                    text = text[:idx]
                else:
                    continue
                s = self.normalize_predicted_step(text, proofs[i])
                try:
                    step = ProofStep(proofs[i], s, strict)
                except InvalidProofStep:
                    continue
                all_proof_steps[i].append(step)
                all_scores[i].append(float(score))

        return all_proof_steps, all_scores

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        if self.stepwise:
            loss = self(
                batch["input_seq_ids"], batch["input_seq_mask"], batch["output_seq_ids"], batch["sentence_span_list"], batch["sentence_current_label"], batch["proof"]
            )
            self.log("loss_train", loss, on_epoch=True, sync_dist=True)

            self.num_step += 1

        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log("loss_train", loss, on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("test", batch, batch_idx)

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("test", outputs)

    def val_test_step(self, split: str, batch: Batch, batch_idx: int) -> Tuple[Any]:
        if self.stepwise:
            proof_pred, score = self.generate_stepwise_proof(batch["proof"], batch_idx)
        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log(f"loss_{split}", loss, sync_dist=True)
            proof_pred, score = self.generate_entire_proof(batch["input_seq"])

        if self.dataset == "entailmentbank":
            return proof_pred, score, batch["proof"]
        else:
            return (
                proof_pred,
                score,
                batch["proof"],
                batch["answer"],
                batch["depth"],
                batch["all_proofs"],
            )

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        results = []

        for out in outputs:
            if self.dataset == "entailmentbank":
                for proof_pred, score, proof in zip(*out):
                    results.append(
                            {
                                "proof_pred": proof_pred,
                                "score": score,
                                "hypothesis": proof.hypothesis,
                                "context": proof.context,
                                "proof_gt": proof.proof_text,
                            }
                        )
            else:
                for proof_pred, score, proof, answer, depth, all_proofs in zip(*out):
                    results.append(
                        {
                            "answer": answer,
                            "depth": depth,
                            "all_proofs": all_proofs,
                            "proof_pred": proof_pred,
                            "score": score,
                            "hypothesis": proof.hypothesis,
                            "context": proof.context,
                            "proof_gt": proof.proof_text,
                        }
                    )

        assert self.trainer is not None
        json_path = os.path.join(self.trainer.log_dir, f"results_{split}_{self.current_epoch}.json")
        json.dump(results, open(json_path, "wt", encoding='utf-8'))
        if self.dataset == "entailmentbank":
            tsv_path = os.path.join(self.trainer.log_dir, f"results_{split}_{self.current_epoch}.tsv")
            with open(tsv_path, "wt", encoding='utf-8') as oup:
                for r in results:
                    proof = r["proof_pred"].strip()
                    if not proof.endswith(";"):
                        proof += ";"
                    try:
                        oup.write(f"$proof$ = {proof}\n")
                    except:
                        # oup.write(f"$proof$ = INVALID_PROOF")  
                        oup.write(f"$proof$ = sent1 -> hypothesis")  
            print(f"Validation results saved to {json_path} and {tsv_path}")

        if self.dataset == "entailmentbank" and results[0]["proof_gt"] != "":
            em, f1 = evaluate_entailmentbank(results, eval_intermediates=False)
            re_acc = {
                'epoch': self.current_epoch,
                'em': em,
                'f1': f1
            }
            json_path = os.path.join(self.trainer.log_dir, f"results_step_{split}.json")
            with open(json_path, 'a', encoding='utf-8') as f:
                # Use json.dump() with indent=4 to write with indentation
                json.dump(re_acc, f, indent=4)
                f.write('\n')  # Add a newline to separate each result
            # json.dump(re_acc, open(json_path, "a", encoding='utf-8'))
            for k, v in em.items():
                self.log(f"ExactMatch_{k}_{split}", v, on_step=False, on_epoch=True)
            for k, v in f1.items():
                self.log(f"F1_{k}_{split}", v, on_step=False, on_epoch=True)

        elif self.dataset == "ruletaker":
            answer_accuracies, proof_accuracies = evaluate_ruletaker(results)
            for k in answer_accuracies.keys():
                self.log(
                    f"Accuracy_answer_{k}_{split}",
                    answer_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Accuracy_proof_{k}_{split}",
                    proof_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())  # type: ignore
                // self.trainer.accumulate_grad_batches
            )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )
