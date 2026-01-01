import pandas as pd
from dataclasses import dataclass
from ranx import Qrels, Run, evaluate
from typing import List, Dict, Union, Optional

GroundTruthQrels = Dict[str, Dict[str, float]]
PredictedRankings = Dict[str, List[str]]
ScoredRun = Dict[str, Dict[str, float]]


def build_perfect_run_from_qrels(
    qrels_dict: Dict[str, Dict[str, float]],
    *,
    tie_breaker: str = "doc_id",  # deterministic tie-breaker among equal-gain docs
) -> Dict[str, List[str]]:
    """
    Returns:
      run_ranked: Dict[qid, List[pid]] in perfect order by gain desc.
    """
    run_ranked: Dict[str, List[str]] = {}

    for qid, doc_gains in qrels_dict.items():
        items = list(doc_gains.items())  # (pid, gain)

        if tie_breaker == "doc_id":
            # sort by gain desc, then doc_id asc
            items.sort(key=lambda x: (-float(x[1]), str(x[0])))
        else:
            # gain desc only (may be nondeterministic across runs due to dict ordering)
            items.sort(key=lambda x: -float(x[1]))

        run_ranked[str(qid)] = [str(pid) for pid, _ in items]

    return run_ranked


@dataclass(frozen=True)
class SearchEvaluator:
    """
    Evaluates retrieval quality using ranx.

    Key concepts:
      - ground_truth_qrels (Qrels):
          {query_id: {doc_id: relevance_gain}}
        This is HUMAN-LABELED truth (graded). Used directly for NDCG.

      - predicted_rankings (Run):
          {query_id: [doc_id_1, doc_id_2, ...]}
        This is SYSTEM output (BM25 / HNSW / hybrid). Only the ordering matters.

    Optional:
      - If you want Recall/MRR to be binary, set binary_threshold:
          gains >= threshold -> relevant (1), else nonrelevant (0).
        NDCG is still computed on the graded qrels (recommended).
    """

    ks: List[int]
    default_run_score_mode: str = "reciprocal_rank"

    def __post_init__(self):
        if not self.ks:
            raise ValueError("ks must be non-empty.")
        if any((not isinstance(k, int)) or k <= 0 for k in self.ks):
            raise ValueError("All K values must be positive integers.")
        object.__setattr__(self, "ks", sorted(set(self.ks)))

    def _metric_names(self) -> List[str]:
        names: List[str] = []
        for k in self.ks:
            names.extend([f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"])
        return names

    # ---------------------------
    # Normalization / validation
    # ---------------------------
    @staticmethod
    def normalize_ground_truth_qrels(raw: Dict[str, Dict[str, Union[int, float, str]]]) -> GroundTruthQrels:
        """
        Ensures:
          - query_id/doc_id are strings
          - gains are floats
        """
        out: GroundTruthQrels = {}
        for qid, doc_gains in raw.items():
            qid_s = str(qid)
            out[qid_s] = {}
            for doc_id, gain in (doc_gains or {}).items():
                out[qid_s][str(doc_id)] = float(gain)
        return out

    @staticmethod
    def binarize_qrels(
        ground_truth_qrels: GroundTruthQrels,
        *,
        threshold: float,
        keep_all_queries: bool = True,
    ) -> GroundTruthQrels:
        """
        Convert graded qrels -> binary qrels.

        If keep_all_queries=True (recommended):
        - queries with no docs >= threshold are kept with an empty dict.
        - avoids ranx key mismatch errors.
        """
        thr = float(threshold)
        out: GroundTruthQrels = {}
        for qid, doc_gains in ground_truth_qrels.items():
            kept = {doc_id: 1.0 for doc_id, g in doc_gains.items() if float(g) >= thr}
            if kept or keep_all_queries:
                out[qid] = kept
        return out

    # ---------------------------
    # Run construction
    # ---------------------------
    @staticmethod
    def build_run_from_ranked_lists(
        predicted_rankings: PredictedRankings,
        *,
        score_mode: str = "reciprocal_rank",
    ) -> Run:
        """
        ranx.Run expects:
          {qid: {doc_id: score}}
        If you only have a ranked list, assign synthetic scores by rank.
        """
        run_dict: ScoredRun = {}
        for qid, docs in predicted_rankings.items():
            if not docs:
                run_dict[str(qid)] = {}
                continue
            scored: Dict[str, float] = {}
            n = len(docs)
            for i, doc_id in enumerate(docs):
                rank = i + 1
                if score_mode == "reciprocal_rank":
                    scored[str(doc_id)] = 1.0 / rank
                elif score_mode == "linear":
                    scored[str(doc_id)] = float(n - i)
                elif score_mode == "constant":
                    scored[str(doc_id)] = 1.0
                else:
                    raise ValueError("score_mode must be one of {'reciprocal_rank','linear','constant'}.")
            run_dict[str(qid)] = scored
        return Run(run_dict)

    @staticmethod
    def build_run_from_scored_dict(scored_run: ScoredRun) -> Run:
        norm: ScoredRun = {str(q): {str(d): float(s) for d, s in docs.items()} for q, docs in scored_run.items()}
        return Run(norm)

    def evaluate_rankings(
        self,
        *,
        ground_truth_qrels: Dict[str, Dict[str, Union[int, float, str]]],
        predicted_rankings: PredictedRankings,
        binary_threshold: Optional[float] = None,
        score_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Easiest path when your system returns ranked doc_id lists.
        """
        if score_mode is None:
            score_mode = self.default_run_score_mode

        gt = self.normalize_ground_truth_qrels(ground_truth_qrels)
        run_obj = self.build_run_from_ranked_lists(predicted_rankings, score_mode=score_mode)

        # graded metrics
        if binary_threshold is None:
            scores = evaluate(Qrels(gt), run_obj, self._metric_names(), make_comparable=True)
            return (
                pd.DataFrame([{
                    "metric": m.split("@")[0],
                    "k": int(m.split("@")[1]),
                    "score": float(v)
                } for m, v in scores.items()])
                .sort_values(["metric", "k"])
                .reset_index(drop=True)
            )

        # NDCG graded
        ndcg_scores = evaluate(Qrels(gt), run_obj, [f"ndcg@{k}" for k in self.ks], make_comparable=True)

        # Recall/MRR binary
        bin_gt = self.binarize_qrels(gt, threshold=binary_threshold, keep_all_queries=True)
        rm_scores = evaluate(
            Qrels(bin_gt),
            run_obj,
            [f"recall@{k}" for k in self.ks] + [f"mrr@{k}" for k in self.ks],
            make_comparable=True,
            )
        rows = []
        for k in self.ks:
            rows.append({"metric": "recall", "k": k, "score": float(rm_scores.get(f"recall@{k}", 0.0))})
            rows.append({"metric": "mrr", "k": k, "score": float(rm_scores.get(f"mrr@{k}", 0.0))})
            rows.append({"metric": "ndcg", "k": k, "score": float(ndcg_scores.get(f"ndcg@{k}", 0.0))})

        return pd.DataFrame(rows).sort_values(["metric", "k"]).reset_index(drop=True)
    
    
if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    
    project_dir = Path(os.getcwd())
    data_dir = project_dir / "Data" / "PROCESSED"
    
    with open(data_dir / "test_qrels.json", "r", encoding="utf-8") as f:
        ground_truth_test_qrels = json.load(f)
    
    predicted_rankings = build_perfect_run_from_qrels(ground_truth_test_qrels)
     
    evaluator_pipeline = SearchEvaluator(ks=[5, 10, 20])
    
    summary = evaluator_pipeline.evaluate_rankings(
        ground_truth_qrels=ground_truth_test_qrels,
        predicted_rankings=predicted_rankings,
        binary_threshold=2.0
    )
    print(summary)
    
    