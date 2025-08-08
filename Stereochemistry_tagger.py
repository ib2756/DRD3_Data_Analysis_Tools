# ==========================================================================================
# Script: SDF Preprocessor with Stereochemistry & Conformer Tagging (+ Optional SDF Export)
#
# DESCRIPTION:
# This script processes an SDF file exported from Schrödinger Maestro’s LigPrep workflow
# (supports `.sdf`, `.sdf.gz`, and `.sdfgz` formats). It automatically:
#
#   1. **Standardizes compound names**:
#      - Converts all molecule titles (the `_Name` property) to uppercase for consistency.
#
#   2. **Detects stereochemistry**:
#      - Uses RDKit to identify defined chiral centers.
#      - Appends stereochemistry tags to molecule titles if chirality is present.
#        Example: `CHEMBL123` → `CHEMBL123_4R_8S`
#
#   3. **Assigns conformer identifiers**:
#      - Groups molecules with the same base name and stereochemistry.
#      - Adds conformer tags `_CONF1`, `_CONF2`, … for duplicates to distinguish them.
#        Example: `CHEMBL123_4R_8S_CONF1`, `CHEMBL123_4R_8S_CONF2`
#
#   4. **Exports tagged metadata to CSV**:
#      - Outputs a table where the first column is the updated `Title` and
#        subsequent columns contain all molecule properties present in the SDF.
#
#   5. **(Optional)** Exports an updated SDF:
#      - Writes a new `.sdf` file with the updated, fully tagged molecule titles.
#
# WHY USE THIS SCRIPT:
# LigPrep often generates multiple stereoisomers or conformers of the same compound
# without clearly labeling them in the `Title` field. This script ensures:
#   - Clear stereochemistry annotations in the title.
#   - Unique naming for multiple conformers.
#   - Easy downstream filtering and analysis (e.g., docking score selection per isomer).
#
# INPUT:
#   - Any valid `.sdf`, `.sdf.gz`, or `.sdfgz` file.
#   - Molecules may come from LigPrep, QikProp, Glide, or any other SDF-generating workflow.
#     QikProp properties are NOT required — the script simply preserves all properties
#     already stored in the file.
#
# OUTPUT:
#   - CSV file: `<output_name>.csv` containing:
#         Title (updated with stereochemistry and conformer tags)
#         + all original SDF property fields (as available)
#   - (Optional) Updated `.sdf` file with tagged molecule titles.
#
# DEPENDENCIES:
#   - RDKit
#   - pandas
#
# USAGE EXAMPLE:
#   $ Stereochemistry_tagger.py
#       Enter the full path to your SDF file (.sdf, .sdf.gz, .sdfgz): /path/to/file.sdfgz
#       Enter output CSV filename (without extension): tagged_output
#       Also save updated SDF file with tagged titles? [y/n]: y
#
# AUTHOR:
#   In Young Bae (with ChatGPT, OpenAI)
# ==========================================================================================

from rdkit import Chem
import pandas as pd
import os
import gzip
import shutil
from collections import defaultdict

# === Input & normalization ======================================================
# Prompt for an SDF-like file. We accept plain SDF (.sdf), gzip-compressed SDF (.sdf.gz),
# and Schrödinger's single-extension compressed variant (.sdfgz).
# Normalize Windows-style backslashes and optional surrounding quotes from copy/paste.
sdf_path = input("Enter the full path to your SDF file (.sdf, .sdf.gz, .sdfgz): ").strip().replace("\\", "/").strip('"')

# If the file is a .sdfgz (LigPrep often exports this), make a byte-for-byte copy to .sdf.gz
# so RDKit can read it with a standard gzip stream. We keep the original and work on the copy.
if sdf_path.endswith(".sdfgz"):
    new_path = sdf_path[:-6] + ".sdf.gz"
    shutil.copyfile(sdf_path, new_path)
    sdf_path = new_path
    print(f"🛠 Converted .sdfgz to .sdf.gz: {sdf_path}")

# === Load molecules =============================================================
# Use RDKit suppliers to stream molecules. removeHs=False preserves explicit Hs from
# upstream tools (useful if properties or valence checks were done with Hs present).
# Note: we filter out None in case of parsing failures.
if sdf_path.endswith(".gz"):
    with gzip.open(sdf_path, 'rb') as gzfile:
        suppl = Chem.ForwardSDMolSupplier(gzfile, removeHs=False)
        mols = [mol for mol in suppl if mol is not None]
else:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]

# === Title normalization ========================================================
# Standardize molecule titles to uppercase for consistent grouping and labeling.
# The title lives in the special SDF property "_Name". If it's missing we skip—those
# molecules will be handled later with a fallback name when we write the table.
for mol in mols:
    if mol is not None and mol.HasProp('_Name'):
        mol.SetProp('_Name', mol.GetProp('_Name').upper())

# === Group molecules by (current) title ========================================
# LigPrep/Glide/QikProp pipelines can produce multiple records with the same base name
# (e.g., enumerated stereoisomers or multiple conformers). We first group by title,
# then within each group we will split by stereochemistry and finally tag conformers.
grouped = defaultdict(list)
for mol in mols:
    if mol is not None and mol.HasProp('_Name'):
        grouped[mol.GetProp('_Name')].append(mol)

# === Assign new titles (stereo + conformer tags) ================================
# We'll build a map from the *object identity* of an RDKit Mol (id(mol)) to its new title.
# Using id(mol) avoids collisions if two molecules end up with identical strings.
new_title_map = {}

for title, group in grouped.items():
    # For each molecule in the group, compute its defined chiral centers.
    # includeUnassigned=False ignores centers without an assigned R/S label (avoids "?").
    # useLegacyImplementation=False uses the modern chiral detection in RDKit.
    stereo_tags = []
    for mol in group:
        stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        # Store as a tuple so it can be used as a dict key later (hashable).
        # Each element is (atom_index, 'R' or 'S').
        stereo_tags.append(tuple(stereo))

    # Split the group by stereochemistry signature so conformer indices apply *within*
    # the same stereo assignment. Example: CHEMBL123_4R_8S gets its own CONF numbering.
    stereo_to_mols = defaultdict(list)
    for i, tag in enumerate(stereo_tags):
        stereo_to_mols[tag].append(group[i])

    # Build the final title per molecule:
    #  - If stereo exists, append something like "_1R_5S"
    #  - If multiple records share the same stereo, append "_CONF1", "_CONF2", ...
    #  - If no stereo exists but there are duplicates, append only "_CONF#"
    for stereo_tag, mol_list in stereo_to_mols.items():
        has_stereo = len(stereo_tag) > 0
        if has_stereo:
            # Stereo suffix is built from (atom index + 1) and the configuration (R/S).
            # RDKit uses 0-based atom indices; we present them 1-based for readability.
            suffix = "_".join([f"{idx+1}{conf}" for idx, conf in stereo_tag])

        for i, mol in enumerate(mol_list, start=1):
            if has_stereo:
                new_title = f"{title}_{suffix}_CONF{i}" if len(mol_list) > 1 else f"{title}_{suffix}"
            else:
                new_title = f"{title}_CONF{i}" if len(mol_list) > 1 else title

            new_title_map[id(mol)] = new_title

# === Flatten to a table (DataFrame) ============================================
# For each molecule, update the in-memory title to the new value (so optional SDF export
# will carry the tagged titles), and collect *all* SDF properties into rows. We don't
# assume any specific property set; whatever is present gets preserved.
data = []
for mol in mols:
    if mol is None:
        continue

    # If a molecule had no _Name originally (or wasn't grouped), fall back gracefully.
    new_title = new_title_map.get(id(mol), mol.GetProp('_Name') if mol.HasProp('_Name') else "UNKNOWN")
    mol.SetProp('_Name', new_title)  # ensure the writer later uses the tagged title

    row = {'Title': new_title}
    for prop in mol.GetPropNames():
        row[prop] = mol.GetProp(prop)
    data.append(row)

df = pd.DataFrame(data)

# Ensure 'Title' is the first column for readability in the CSV.
cols = df.columns.tolist()
cols.remove('Title')
df = df[['Title'] + cols]

# === Write CSV =================================================================
# Output CSV sits next to the input file by default. Filename is user-provided (or default).
output_name = input("Enter output CSV filename (without extension): ").strip()
if not output_name:
    output_name = "tagged_sdf_export"
output_dir = os.path.dirname(sdf_path)
csv_path = os.path.join(output_dir, f"{output_name}.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Tagged metadata saved to CSV: {csv_path}")

# === Optional: write a new SDF with updated titles =============================
# If requested, serialize molecules with their updated _Name fields so downstream tools
# (e.g., docking/analysis scripts) see the same stereo/CONF tags as in the CSV.
save_sdf = input("Also save updated SDF file with tagged titles? [y/n]: ").strip().lower().startswith("y")
if save_sdf:
    sdf_out_path = os.path.join(output_dir, f"{output_name}.sdf")
    writer = Chem.SDWriter(sdf_out_path)
    for mol in mols:
        writer.write(mol)
    writer.close()
    print(f"✅ Updated SDF file saved to: {sdf_out_path}")
