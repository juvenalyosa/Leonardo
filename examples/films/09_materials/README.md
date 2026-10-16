# Film 09 - Materials: building surfaces, a protein on three of them, and the orbitals at the contact

Replay pack for the materials film (YouTube 16:9 and Instagram 9:16 versions). Every number in the film
comes from this pipeline, run through Leonardo's own tools:

1. `build_material` - an icosahedral Au309 nanoparticle, Pt(111), graphene, hydroxylated silica (001)
   (ClayFF) and rutile TiO2 (110).
2. `load_structure_file` + `build_protein_surface_system` (x3) - hen egg-white lysozyme (PDB 1AKI) placed
   3 A above Au(111), graphene and silica in water with ions, minimized; contacts and the protein-surface
   Lennard-Jones contact energy. These are energies of the minimized placement, not binding free energies.
3. `load_structure_file` + `cut_qm_cluster` (Arg68 on gold, 4.0 A, hydrogen-capped cut bonds) + `run_mopac`
   (PM7, COSMO water, single point at the adsorbed geometry) + `show_orbitals` (HOMO, LUMO). These steps
   start from `lyso_on_Au_minimized_site.pdb`, the lysozyme-gold structure minimized for the film (protein +
   gold atoms within 12 A of Arg68), so they reproduce the film's numbers exactly.

Why a recorded structure for the QM part: the minimization in step 2 is not bit-reproducible on parallel
platforms (OpenCL or multi-threaded CPU), and the frontier orbitals of a small cluster are very sensitive to
the contact geometry. In four independent minimizations the cluster's HOMO ranged from -5.9 to -8.2 eV.
Read the orbital energies as properties of this one adsorbed geometry, not of lysozyme on gold in general.

```
python -m leonardo.replay verify examples/films/09_materials
```

About 2 minutes. The minimization runs on the GPU and is not bit-reproducible: adsorption energies vary by
roughly 10 kJ/mol between runs (tolerance 25 kJ/mol); structures, atom counts and the QM results are exact.
Recorded values: Au(111) -225.0, graphene -71.2, silica -19.7 kJ/mol; HOMO -8.17 eV (92 % on the gold
atoms), LUMO -0.55 eV (51 % gold, 49 % Arg68 guanidinium and neighbours).

The film scripts: `scripts/materials_film_molecules.py` (PyMOL) and `scripts/materials_film_compose.py`.
Music: "Ouroboros" / "Exhilarate" by Kevin MacLeod (incompetech.com), CC BY 4.0.
