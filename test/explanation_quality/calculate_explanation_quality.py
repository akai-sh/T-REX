import argparse
import json
import re
import sys
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the explanation-quality metrics in Figure 8 from "
            "single-step execution results."
        )
    )
    parser.add_argument(
        "--result_path",
        required=True,
        help="Single-step result JSONL containing output and model_output",
    )
    parser.add_argument(
        "--bert_model_path",
        "--bert_model",
        dest="bert_model_path",
        default="FacebookAI/roberta-large",
        help=(
            "Hugging Face model name or local checkpoint directory used by "
            "BERTScore"
        ),
    )
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--device", help="BERTScore device, such as cuda:0 or cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scores_output",
        help="Optional JSONL output for per-record similarity scores",
    )
    parser.add_argument(
        "--skip_invalid_json",
        action="store_true",
        help="Report and skip malformed JSONL records",
    )
    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Calculate only ROUGE-L and BLEU-4",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    return args


def sample_key(item):
    return item.get("source_id"), item.get("sub_id"), item.get("id")


def read_jsonl(path, skip_invalid=False):
    items = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if not skip_invalid:
                    raise ValueError(
                        f"Invalid JSON in {path!r} at line {line_number}"
                    ) from exc
                print(
                    f"Skipping invalid JSON at {path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
    if not items:
        raise RuntimeError(f"No valid records found in {path!r}.")
    return items


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def main():
    args = parse_args()
    items = read_jsonl(
        args.result_path,
        skip_invalid=args.skip_invalid_json,
    )

    pairs = []
    for item in items:
        key = sample_key(item)
        reference = item.get("output")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Missing or invalid output for sample {key}")
        candidate = item.get("model_output")
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"Missing or invalid model_output for sample {key}")
        pairs.append((key, candidate, reference))
        if args.limit is not None and len(pairs) >= args.limit:
            break

    if not pairs:
        raise RuntimeError("No valid explanation pairs were found.")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    rouge_scores = []
    bleu_scores = []
    for _, candidate, reference in pairs:
        rouge_scores.append(rouge.score(reference, candidate)["rougeL"].fmeasure)
        bleu_scores.append(
            sentence_bleu(
                [tokenize(reference)],
                tokenize(candidate),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )

    bert_f1_scores = None
    if not args.skip_bertscore:
        from bert_score import score as bert_score

        candidates = [candidate for _, candidate, _ in pairs]
        references = [reference for _, _, reference in pairs]
        _, _, bert_f1 = bert_score(
            candidates,
            references,
            model_type=args.bert_model_path,
            num_layers=17,
            batch_size=args.bert_batch_size,
            device=args.device,
            verbose=True,
            rescale_with_baseline=False,
        )
        bert_f1_scores = bert_f1.tolist()

    print(f"Records: {len(pairs)}")
    print(f"ROUGE-L: {sum(rouge_scores) / len(rouge_scores):.4f}")
    print(f"BLEU-4: {100 * sum(bleu_scores) / len(bleu_scores):.2f}")
    if bert_f1_scores is not None:
        print(
            "BERTScore F1: "
            f"{100 * sum(bert_f1_scores) / len(bert_f1_scores):.2f}"
        )

    if args.scores_output:
        output_path = Path(args.scores_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as destination:
            for index, ((key, _, _), rouge_l, bleu_4) in enumerate(
                zip(pairs, rouge_scores, bleu_scores)
            ):
                result = {
                    "source_id": key[0],
                    "sub_id": key[1],
                    "id": key[2],
                    "rouge_l": rouge_l,
                    "bleu_4": 100 * bleu_4,
                }
                if bert_f1_scores is not None:
                    result["bertscore_f1"] = 100 * bert_f1_scores[index]
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Per-record scores saved to: {output_path}")


if __name__ == "__main__":
    main()
