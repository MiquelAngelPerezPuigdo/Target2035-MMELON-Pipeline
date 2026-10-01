# OOD assessment after 25/50 validation hits

The user reported 25 hits in 3 chemical clusters for round03 G (p = 2.0888282834863325e-05), zero for H, and 1 hit in 1 cluster for I (p = 0.052011861818003435). There are 39 validations remaining before round04.

The known-ligand ranking was defined before this result. It uses pre-existing public crystal ligands and no individual activity labels from the 25-hit set. The result therefore supports those ligand priors on validation. It does not establish transfer to chemically different test compounds, nor that the DEL model is strong: the winning formula gives 95% weight to ligand similarity and 5% to the DEL-trained ensemble. Future tuning on repeated validation outcomes can introduce additional selection overfitting.

The successful method, reference definitions and trained model files have been archived under models/frozen_round03 with hashes. The original method is preserved unchanged.

## Unlabelled coverage audit

The frozen method was applied to the provided validation and test SMILES without accessing test activity labels or making a test submission.

| Measure | Validation | Test |
|---|---:|---:|
| Library size | 244,328 | 184,632 |
| Maximum reference-ligand ECFP similarity | 0.5625 | 0.3611 |
| Compounds with similarity >= 0.35 | 14 | 3 |
| Compounds with similarity >= 0.40 | 4 | 0 |
| Median similarity in frozen top 50 | 0.3216 | 0.3034 |
| Quinazoline-containing compounds in frozen top 50 | 35 | 14 |
| Top-50 preference for ligand 21 / ligand 47 | 36 / 14 | 19 / 31 |

This supports a chemical-coverage concern: the test set has fewer close reference analogues and shifts toward the other ligand. The smaller median shift also matters: one should not infer failure from the maximum alone. These are structural measurements, not predicted hit counts. Individual hit identities are unknown, so we cannot attribute all 25 hits to any particular core or ligand.

Our Murcko scaffold counts are not the organizers' chemical-series labels. Three experimental hit clusters provide much less evidence of breadth than 25 hit molecules considered in isolation. The organizers confirm that each split contains over 50 series or singleton hits, so three clusters are not an inherent ceiling of the dataset.

Sources: [challenge overview](https://www.synapse.org/Synapse:syn75349604/wiki/641044), [organizer clarification on series](https://www.synapse.org/Synapse:syn75349604/discussion/threadId=14714), [public ligand sources](https://cache-challenge.org/challenges/finding-selective-inhibitors-of-phosphoglycerate-kinase-2).

## Next validation experiments

Round04 contains three deliberately harder tests, with all previously submitted IDs excluded:

- J: new candidates from ligand 47, which is more prominent in the unlabelled test ranking.
- K: new core structures, explicitly excluding quinazolines for this diagnostic experiment.
- L: candidates with reference similarity <= 0.36 and similarity to every original G prediction <= 0.40, ranked using the frozen winning score.

These filters are imperfect stress tests; they do not reproduce the organizers' unknown test hit clusters. Lower validation counts are possible and informative. We will compare hits and chemical clusters. No test submission should be spent solely on the assumption that 25/50 validation transfers unchanged. Both test submissions remain unused by this workflow.
