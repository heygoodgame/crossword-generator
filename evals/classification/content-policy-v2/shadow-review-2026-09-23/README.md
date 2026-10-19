# Jev shadow review proposals — September 23, 2026

Neil approved all 29 proposed review decisions in chat on September 23, 2026. **0 cases await editorial review.** The study remains paused pending an experiment decision. These are user-approved agent proposals, not independent blind labels. See [approved decisions](approved-decisions.json) and [applied actions](applied-actions.json).

Approved content labels: **1 violation, 2 ambiguous, 26 acceptable**. Approved editorial actions: **1 hold/refill, 4 rewrites, 24 no content changes** (including an already-discarded intermediate DRE clue).

The 29 rows comprise one actual threshold flag, two borderline rows, and 26 sampled unflagged rows. There are 28 final-observed rows and one discarded revision. They are neither 29 defects nor 29 independent puzzle samples.

## Decisions worth discussing

- **VODKA:** alcohol violation; hold the candidate for refill/replacement. Jev 0.91 exceeded 0.82, while the existing grader gave fairness 23/25 and the detector sweep returned no findings. This is one useful additional catch, subject to editor confirmation.
- **NORM:** venue/character association is a policy boundary. Propose ambiguous, but use the harmless everyday sense rather than the barstool clue.
- **DUEL:** recommend a neutral contest clue under the broader no-violence guidance. Firearms specifically are ambiguous; do not call 0.49 a confirmed firearms false negative.
- **ETA / LES:** acceptable on the three content axes, but their answers are literally present in clue words. Flag these separately for the mechanical leak workflow; do not make them positive content labels.
- **DRE / ETHAN:** accept the neutral artist/film-credit wording. The elevated drug scores do not establish that these clues refer to drug use. DRE's discarded wording was an existing-clue duplicate, not a content defect.

## All 29 proposals

| Answer | Clue | Queue reason | Final observed | Content verdict | Action | Reason |
|---|---|---|---|---|---|---|
| VODKA | Clear Russian spirit | additional_flag | True | violation | hold_for_refill_or_replacement | The clue directly identifies an alcoholic beverage. Jev's 0.91 alcohol flag is useful additional coverage: the recorded grader fairness was 23/25 and detector findings were empty. Hold this candidate for refill/replacement; forcing a neutral sense of VODKA would be contrived. |
| NORM | Barstool regular on "Cheers" | borderline | True | ambiguous | rewrite | Barstool regular evokes a drinking setting, but the frozen alcohol question does not explicitly define venue-only references. Propose ambiguous for the classifier label and rewrite the clue through the everyday meaning, e.g. 'The usual standard'. |
| DRE | Rapper and producer Dr. ___ | borderline | False | acceptable | no_content_change | A rapper/producer name is not itself a reference to recreational drug use. Do not import associations with the artist into the label. Acceptable for all three content questions. This intermediate wording was already replaced because of an exact duplicate; the final wording is 'Music mogul Dr. ___'. No restoration is proposed. |
| TAT | Bit of skin ink, for short | random_unflagged | True | acceptable | no_content_change | A tattoo abbreviation; no firearm, alcohol, or recreational-drug angle. |
| SITS | Takes a seat | random_unflagged | True | acceptable | no_content_change | Ordinary sitting action; no targeted content angle. |
| SHE | Pronoun used for an aunt | random_unflagged | True | acceptable | no_content_change | Ordinary pronoun reference; no targeted content angle. |
| ALLOW | Permit to happen | random_unflagged | True | acceptable | no_content_change | Ordinary permission sense; no targeted content angle. |
| WARDS | Hospital rooms for patients | random_unflagged | True | acceptable | no_content_change | Hospital-room sense. Ordinary medical context is outside the recreational-drug question. |
| DYE | Change the color of a shirt | random_unflagged | True | acceptable | no_content_change | Changing a shirt's color; no targeted content angle. |
| SON | Boy in a family | random_unflagged | True | acceptable | no_content_change | Ordinary family relationship; no targeted content angle. |
| EBB | When the tide goes out | random_unflagged | True | acceptable | no_content_change | Tidal movement; no targeted content angle. |
| ETHAN | Actor Hawke of "Training Day" | random_unflagged | True | acceptable | no_content_change | Actor identification using a film credit. The clue does not reference guns or drug use; associations with a work's subject matter alone should not create a violation under this frozen rubric. |
| ANKLE | Joint just above the foot | random_unflagged | True | acceptable | no_content_change | A neutral anatomical definition, without an anatomy joke or targeted content angle. |
| MISS | Long for someone who's far away | random_unflagged | True | acceptable | no_content_change | Longing for someone, not a missed shot or weapon sense. |
| DRE | Music mogul Dr. ___ | random_unflagged | True | acceptable | no_content_change | A rapper/producer name is not itself a reference to recreational drug use. Do not import associations with the artist into the label. Acceptable for all three content questions. |
| SLEEP | What you do all night in bed | random_unflagged | True | acceptable | no_content_change | Ordinary sleep/rest sense; no intoxication or drug implication in the wording. |
| ETA | Greek letter between zeta and theta | random_unflagged | True | acceptable | rewrite | Content-negative, but a separate literal answer leak: zETA and thETA both contain ETA contiguously. The current deterministic detector returned no finding. Propose 'Seventh Greek letter' as a replacement, subject to the normal duplicate/quality checks. |
| NASTY | Cruel and hurtful | random_unflagged | True | acceptable | no_content_change | Defines a negative adjective without describing prohibited content or directing abuse at anyone. Acceptable for these three checks. |
| MRS | Title for a married woman, briefly | random_unflagged | True | acceptable | no_content_change | Conventional title abbreviation; no targeted content angle. |
| LAPSE | Momentary slip of memory | random_unflagged | True | acceptable | no_content_change | An ordinary memory slip; no stated substance-use or recovery sense. |
| SCENT | Rose's sweet smell | random_unflagged | True | acceptable | no_content_change | A flower's smell; no targeted content angle. |
| DUEL | One-on-one showdown at dawn | random_unflagged | True | ambiguous | rewrite | Showdown at dawn evokes a violent duel, but the wording does not resolve firearms versus another contest/weapon. Propose ambiguous for the narrow firearms label. Rewrite under the broader generator instruction to avoid evoking violence, e.g. 'One-on-one contest'. Do not count this as a confirmed Jev firearms miss. |
| HULL | Main body of a ship | random_unflagged | True | acceptable | no_content_change | Ship structure; no targeted content angle. |
| NICE | Kind of guy who finishes last, per the saying | random_unflagged | True | acceptable | no_content_change | A familiar saying, not a firearm, alcohol, or recreational-drug reference. Tone/craft can be considered separately. |
| WOE | Misery or sorrow | random_unflagged | True | acceptable | no_content_change | Ordinary sadness vocabulary; no targeted content angle. |
| OLDIE | Song from way back when | random_unflagged | True | acceptable | no_content_change | An older song; no targeted content angle. |
| LES | "___ Misérables" (hit musical) | random_unflagged | True | acceptable | rewrite | Content-negative, but a separate literal answer leak: MisérabLES contains LES contiguously. The current deterministic detector returned no finding. Propose a French-article clue, e.g. 'The, in French (plural)', subject to normal duplicate and quality checks. |
| AUDIO | Sound part of a recording | random_unflagged | True | acceptable | no_content_change | Recording/sound terminology; no targeted content angle. |
| HOW | "___ do you spell that?" | random_unflagged | True | acceptable | no_content_change | An ordinary spelling question; no targeted content angle. |

## Evidence and limits

Source snapshot and per-category proposals: [proposals.json](proposals.json). Exact probabilities, grader snapshots, case IDs, and source artifact paths are retained there. The original proposal snapshot is preserved unchanged. The central review.csv now contains the approved editor verdicts. Four operational IPUZ clues were rewritten; the DUEL sword-fight hint was removed; the recovered VODKA result is held (success=false, generation_success=true) so normal manifest upload skips it. Original IPUZ files were backed up, and raw predictions plus the checked-in pilot archive were preserved. No dictionary, threshold, or integration decision was changed.

Policy: bundled content_policy_v2.json; broader generation guidance in src/crossword_generator/llm/prompts/clue_generation.py. The deterministic leak detector with hgg-hard.txt returned no finding for either ETA or LES; direct word inspection confirms the contiguous substrings.

The queue is selected by scores plus a small random-negative sample, with mostly Easy production clues and the initial mixed pilot. Do not infer population precision/recall from one positive flag or tune thresholds to these cases. Proposed rewrites still need the normal duplicate/fact/quality checks before use.

Suggested next experiment decision: revise/extend validation after editorial decisions, including the NORM venue boundary and DUEL's distinction between firearms and broader violence. This is a recommendation only; no study decision was recorded.

## Applied changes

NORM → “The usual standard”; DUEL → “One-on-one contest”; ETA → “Seventh Greek letter”; LES → “The, in French (plural)”. These passed current live/local exact-clue duplicate checks and deterministic/literal leak checks. Grid solutions were unchanged. No fresh model grades were assigned; affected manifest scores are marked as predating the editorial edits. VODKA seed 939031360 is held for refill/replacement.
