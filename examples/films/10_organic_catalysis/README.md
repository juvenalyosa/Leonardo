# Film 10 - Organic chemistry and catalysis: aspirin in infrared, a keto-enol tautomerization with and without a water catalyst

Replay pack for the organic chemistry & catalysis film (YouTube 16:9 and Instagram 9:16 versions). Every
number in the film comes from this pipeline, run through Leonardo's own tools (`run_psi4`, B3LYP-D3(BJ)/def2-SVP,
gas phase):

1. Aspirin (`aspirin.mol`, RDKit/MMFF start): optimization + harmonic frequencies (`stationary`), the IR
   spectrum and the normal modes the film animates (strongest C=O and O-H stretches; unscaled).
2. Acetaldehyde -> vinyl alcohol, uncatalysed: keto minimum, transition state (`ts`, one imaginary mode) and
   enol minimum. The proton shifts from carbon to oxygen through a strained four-membered ring.
3. The same reaction relayed by one water molecule: keto...H2O complex, the six-membered-ring TS (two protons
   move in concert: C-H to the water oxygen, a water proton to the carbonyl oxygen) and enol...H2O complex.

Barriers are E(TS) - E(keto) from the electronic energies (no zero-point or thermal corrections, gas phase);
they are not free energies. The film moves the atoms between the computed stationary points (keto, TS, enol);
the drawn path is an interpolation, not an IRC.

Start structures: the TS guesses come from MOPAC PM7 TS searches; the minima start from the structures reached
by displacing each B3LYP TS along its imaginary mode and minimizing (so each TS connects the minima used).
Leonardo optimizes all of them again in the pack.

```
python -m leonardo.replay verify examples/films/10_organic_catalysis
```

About 40 minutes on 4 threads (aspirin's finite-difference frequencies take most of it). Psi4 results are
reproducible to well below the tolerance (0.2 mEh). Recorded values: uncatalysed TS 2052i cm-1, barrier
69.5 kcal/mol, enol 11.1 kcal/mol above the keto form; water-relayed TS 1642i cm-1, barrier 37.8 kcal/mol
(31.8 lower), enol complex 9.1 kcal/mol above the keto complex. Aspirin: no imaginary modes; the film shows the
1835 cm-1 C=O stretch and the 3750 cm-1 O-H stretch (harmonic, unscaled).

Leonardo fixes found while building this pack: a TS search now starts from an exact Hessian and, when soft
extra imaginary modes remain, displaces along them (0.5 A) and re-optimizes with tight convergence; the IRC gives
each branch room to settle (geom_maxiter >= 15 x points) and bonds a migrating atom to both partners
(covalent_connect 1.6). For this four-membered keto-enol TS OptKing's IRC still does not follow the reaction
coordinate cleanly, which is why the pack characterizes the stationary points instead.

The film scripts: `scripts/organic_film_data.py` (collects the pack outputs), `scripts/organic_film_molecules.py`
(PyMOL) and `scripts/organic_film_compose.py`. Music: "Getting it Done" / "Rocket" by Kevin MacLeod
(incompetech.com), CC BY 4.0.
