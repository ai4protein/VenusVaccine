import torch
import numpy as np
from biotite.structure.io.pdb import PDBFile
from src.esm.utils.structure.protein_chain import ProteinChain
from src.esm.models.vqvae import StructureTokenEncoder

class StructureFeatureProcessor:
    def __init__(self, device="cpu"):
        self.device = device
        self.encoder = self._load_esm3_encoder()
    
    def _load_esm3_encoder(self):
        model = (
            StructureTokenEncoder(
                d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
            )
            .to(self.device)
            .eval()
        )
        state_dict = torch.load(
            "src/data/weights/esm3_structure_encoder_v0.pth", map_location=self.device
        )
        model.load_state_dict(state_dict)
        return model

    def get_esm3_structure_seq(self, pdb_file):
        chain_ids = self._get_chain_ids(pdb_file)
        chain = ProteinChain.from_pdb(pdb_file, chain_id=chain_ids[0])
        
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.to(self.device)
        residue_index = residue_index.to(self.device)
        
        _, structure_tokens = self.encoder.encode(coords, residue_index=residue_index)
        return structure_tokens.cpu().numpy().tolist()[0], chain.sequence

    @staticmethod
    def _get_chain_ids(pdb_file):
        return np.unique(PDBFile.read(pdb_file).get_structure().chain_id) 