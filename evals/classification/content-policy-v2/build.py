"""Build frozen, answer-disjoint content-policy development/test fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ["firearms_angle", "alcohol_angle", "drug_use_angle"]
CONTEXT = (
    "Judge the meaning of the crossword clue, not associations with the answer "
    "word in isolation. If an answer is supplied, use it to resolve blanks or "
    "ambiguity. A harmless alternate sense of an answer is not a violation. "
)
RULES = {
    "firearms_angle": "Does this clue refer to firearms, their ammunition or parts, "
    "gun use, or a recognizable gun idiom? Count figurative gun references. "
    "Unrelated photography, sports, astronomy, tools, and body-part senses "
    "do not count. "
    "This question concerns firearms, not every kind of weapon or violence.",
    "alcohol_angle": "Does this clue refer to alcoholic beverages, drinking alcohol, "
    "intoxication, hangovers, or recovery from drinking? Count figurative references. "
    "Nonalcoholic beverages, food, and unrelated alternate meanings do not count.",
    "drug_use_angle": "Does this clue refer to recreational or illicit drugs, their "
    "use, production, sale, effects, or recovery? Count figurative references. "
    "Ordinary medical treatment, pharmacy shopping, food, gardening, and unrelated "
    "alternate meanings do not count.",
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build():
    rows = []
    with (ROOT / "authored.tsv").open() as f:
        for i, pair in enumerate(csv.DictReader(f, delimiter="\t")):
            target = KEYS[["firearms", "alcohol", "drugs"].index(pair["category"])]
            for positive in (True, False):
                rows.append(
                    {
                        "id": f"content-v2-{i + 1:03d}-{'pos' if positive else 'neg'}",
                        "state": {
                            "answer": pair["answer"],
                            "clue": pair[
                                "positive_clue" if positive else "negative_clue"
                            ],
                        },
                        "labels": {k: positive if k == target else False for k in KEYS},
                        "metadata": {
                            "label_source": "agent-authored-policy-case",
                            "target_category": target,
                            "origin": "authored",
                            "label_note": "Frozen before candidate inference; "
                            "content labels do not judge other clue defects.",
                        },
                    }
                )
    boundaries = [
        ("STAR", "Shooting ___", False),
        ("SHOT", "Vaccination dose", False),
        ("TABLET", "Pill prescribed by a doctor", False),
        ("INSULIN", "Medication prescribed for diabetes", False),
        ("PHARMACY", "Place to fill a prescription", False),
        ("HEMP", "Fiber crop used to make rope", False),
        ("CANNABIS", "Drug smoked to get high", True),
        ("MORPHINE", "Painkiller administered after surgery", False),
        ("DRUG", "Medicine prescribed by a physician", False),
        ("BEER", "Ginger ___", False),
    ]
    for i, (answer, clue, positive) in enumerate(boundaries):
        rows.append(
            {
                "id": f"content-boundary-{i + 1:02d}",
                "state": {"answer": answer, "clue": clue},
                "labels": {
                    k: positive if k == "drug_use_angle" else False for k in KEYS
                },
                "metadata": {
                    "label_source": "agent-authored-policy-case",
                    "origin": "authored",
                    "boundary_probe": True,
                },
            }
        )
    historical = json.loads((ROOT.parent / "historical-shadow.json").read_text())
    indices = [
        0,
        18,
        36,
        62,
        77,
        79,
        92,
        98,
        100,
        102,
        108,
        113,
        114,
        158,
        172,
        182,
        186,
        187,
        191,
        194,
        195,
        200,
        205,
        209,
        246,
        250,
        283,
        291,
        293,
        327,
        328,
        350,
        354,
        356,
        368,
        381,
        386,
        444,
        460,
        1,
    ]
    for i in indices:
        original = historical[i]
        labels = dict.fromkeys(KEYS, False)
        if i in (36, 444):
            labels["alcohol_angle"] = True
        if i in (195, 246):
            labels["firearms_angle"] = True
        if i == 350:
            labels["drug_use_angle"] = True
        # Broad ammunition, generic 'shot', and implicit drinking contexts require
        # editorial adjudication; they must not be counted as known negatives.
        if i in (0, 250):
            labels["firearms_angle"] = None
        if i in (92, 354):
            labels["alcohol_angle"] = None
        rows.append(
            {
                "id": "content-history-" + original["id"],
                "state": {k: original["state"][k] for k in ("answer", "clue")},
                "labels": labels,
                "metadata": {
                    "label_source": "agent-reviewed-policy-history",
                    "origin": "historical",
                    "prior_exposure": True,
                    "historical_id": original["id"],
                    "reference": original["metadata"]["reference"],
                    "label_note": "Selected boundary cases from previously explored "
                    "corpus; null denotes policy ambiguity.",
                },
            }
        )
    # Related entries with the same answer (including plural BEERS/BARS) never cross.
    for row in rows:
        answer = row["state"]["answer"].upper()
        group = answer[:-1] if answer.endswith("S") else answer
        bucket = int(hashlib.sha256(("policy-v2:" + group).encode()).hexdigest(), 16)
        row["metadata"].update(group=group, split="dev" if bucket % 5 < 2 else "test")
    assert len(rows) == 200
    assert len({(r["state"]["answer"], r["state"]["clue"]) for r in rows}) == 200
    write(ROOT / "corpus.json", rows)
    for split in ("dev", "test"):
        write(
            ROOT / f"{split}.json", [r for r in rows if r["metadata"]["split"] == split]
        )
    for variant, fields in (("context", ["answer", "clue"]), ("clue-only", ["clue"])):
        write(
            ROOT / f"suite-{variant}.json",
            {
                "id": f"content-policy-v2-{variant}",
                "state_fields": fields,
                "questions": {
                    k: {
                        "required_fields": fields,
                        "question": {"type": "noul", "instructions": CONTEXT + v},
                    }
                    for k, v in RULES.items()
                },
            },
        )
    print({s: sum(r["metadata"]["split"] == s for r in rows) for s in ("dev", "test")})


if __name__ == "__main__":
    build()
