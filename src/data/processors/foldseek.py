import subprocess
import os
from Bio import SeqIO

class FoldseekProcessor:
    @staticmethod
    def run_foldseek_commands(pdb_dir):
        temp_dir = "temp"
        fasta_file = "foldseek_seq.fasta"
        
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            subprocess.run(["foldseek", "createdb", pdb_dir, f"{temp_dir}/db"], check=True)
            subprocess.run(["foldseek", "lndb", f"{temp_dir}/db_h", f"{temp_dir}/db_ss_h"], check=True)
            subprocess.run(["foldseek", "convert2fasta", f"{temp_dir}/db_ss", fasta_file], check=True)
            
            foldseek_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
            
            return foldseek_dict
        finally:
            if os.path.exists(temp_dir):
                subprocess.run(["rm", "-rf", temp_dir], check=True)
            if os.path.exists(fasta_file):
                os.remove(fasta_file)