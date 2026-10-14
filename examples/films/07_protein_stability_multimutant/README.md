# Replay pack: film 07, screening disulfide designs

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/07_protein_stability_multimutant
```

It takes about 2 minutes on a laptop. It needs no API key.

This pack runs the same tool as pack 06 (`design_disulfides`, see its README) on protein G B1, but refines only
the 5 best-ranked candidates. It shows how each one does against the same geometric filters.

## Reference result

| Design | S–S | χ3 | Cα shift | Verdict |
|---|---|---|---|---|
| A20C/A26C | 2.04 Å | 85° | 0.20 Å | passes, best |
| A34C/V39C | 2.04 Å | −67° | 0.20 Å | passes |
| A20C/T25C | 2.05 Å | −69° | 0.20 Å | passes |
| Y3C/A26C | 2.04 Å | 79° | 0.19 Å | passes |
| I6C/T53C | 2.04 Å | −62° | 0.19 Å | fails: \|χ3\| below 65° |

The best of these five, A20C/A26C, reduces the pair's relative motion by 41% (network model). Screening more
candidates (pack 06) finds K10C/E56C, whose χ3 is closer to 90°.

These are structural filters, not measured stabilities. The film's scripted run screened a different set of five
designs, chosen by hand.
