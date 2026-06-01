from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider


class EvalCap(COCOEvalCap):
    def __init__(self, coco, coco_res):
        super().__init__(coco, coco_res)

    def evaluate(self):
        img_ids = self.params["image_id"]
        gts = {}
        res = {}
        for img_id in img_ids:
            gts[img_id] = self.coco.imgToAnns[img_id]
            res[img_id] = self.cocoRes.imgToAnns[img_id]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, metric_name in zip(score, scores, method):
                    self.setEval(sc, metric_name)
                    self.setImgToEvalImgs(scs, gts.keys(), metric_name)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Spot-the-Diff captions.")
    parser.add_argument(
        "--gt_json",
        default="gt/spot_total_change_captions_reformat.json",
        help="COCO-style Spot ground-truth JSON.",
    )
    parser.add_argument("--pred_json", required=True, help="Prediction JSON path.")
    parser.add_argument(
        "--output_json",
        default="outputs/qwen25_vl_spot/metrics.json",
        help="Where to save the metrics JSON.",
    )
    return parser.parse_args()


def score_generation(anno_file: str, result_file: str):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = EvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)


def main() -> None:
    args = parse_args()
    metrics = score_generation(args.gt_json, args.pred_json)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    for key in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]:
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
