import sys
import os
import pandas as pd
from tqdm import tqdm
import warnings
from src.data.processors.descriptor_features import DescriptorFeatureProcessor
from src.data.processors.structure_features import StructureFeatureProcessor
from src.data.processors.foldseek import FoldseekProcessor

warnings.filterwarnings("ignore")


def process_pdb_folder(pdb_dir, output_json_file):
    # Initialize processors
    descriptor_processor = DescriptorFeatureProcessor()
    structure_processor = StructureFeatureProcessor()
    
    # Get Foldseek features
    foldseek_dict = FoldseekProcessor.run_foldseek_commands(pdb_dir)
    
    results = []
    for pdb_file in tqdm(os.listdir(pdb_dir)):
        if not pdb_file.endswith(".pdb"):
            continue
            
        pdb_path = os.path.join(pdb_dir, pdb_file)
        name = pdb_file[:-4]
        
        # Get structure features
        esm3_structure_seq, sequence = structure_processor.get_esm3_structure_seq(pdb_path)
        
        # Get other features
        foldseek_seq = foldseek_dict.get(name)
        e_descriptor = descriptor_processor.e_descriptor_embedding(sequence)
        z_descriptor = descriptor_processor.z_descriptor_embedding(sequence)
        
        result = {
            "name": name,
            "aa_seq": sequence,
            "esm3_structure_seq": esm3_structure_seq,
            "foldseek_seq": foldseek_seq,
            "e_descriptor": e_descriptor,
            "z_descriptor": z_descriptor
        }
        results.append(result)

    # Save results
    pd.DataFrame(results).to_json(output_json_file, orient="records", lines=True)
    print("JSON file created successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdb2json.py <pdb_dir> <output_json_file>")
        sys.exit(1)
        
    pdb_dir = sys.argv[1]
    output_json_file = sys.argv[2]
    process_pdb_folder(pdb_dir, output_json_file)
