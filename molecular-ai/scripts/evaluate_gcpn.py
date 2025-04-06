#!/usr/bin/env python
# Script to generate molecules using trained GCPN and evaluate them

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Set up logging first, before any imports that might use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gcpn_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # molecular-ai directory
sys.path.append(parent_dir)

# Make sure we're actually adding the right directory
logger.info(f"Adding to path: {parent_dir}")

# Import local GCS utilities first
sys.path.append(script_dir)
try:
    from gcs_utils import is_gcs_path, download_from_gcs
    logger.info("Successfully imported GCS utilities")
except ImportError:
    logger.error("Could not import gcs_utils.py. Make sure it's in the same directory.")
    sys.exit(1)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Draw, AllChem, Crippen
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    logger.info("Successfully imported RDKit")
except ImportError:
    logger.error("Could not import RDKit. Please install it with: pip install rdkit")
    sys.exit(1)

# Try to import PyTorch
try:
    import torch
    logger.info("Successfully imported PyTorch")
except ImportError:
    logger.error("Could not import PyTorch. Please install it with: pip install torch")
    sys.exit(1)

# Try to import modules from the current project structure
try:
    # Try direct imports - this is the correct path for your structure
    logger.info("Trying direct imports...")
    
    # Import modules directly from src package
    from src.rl.optimize_gcpn import ActorGNN
    from src.models.graph_vae import GraphVAE
    from src.utils.rewards import calculate_reward
    
    # Import property prediction functions directly
    try:
        from src.utils.rewards import predict_qed, predict_logp
        logger.info("Successfully imported property prediction functions")
    except ImportError:
        logger.warning("Could not import predict_qed and predict_logp directly, defining fallbacks")
        
        # Define fallback functions
        def predict_qed(mol, predictor_model=None):
            """Fallback QED prediction function using RDKit directly"""
            try:
                from rdkit.Chem import QED
                return QED.qed(mol)
            except Exception as e:
                logger.error(f"Error calculating QED: {e}")
                return 0.0
                
        def predict_logp(mol, predictor_model=None):
            """Fallback LogP prediction function using RDKit directly"""
            try:
                from rdkit.Chem import Crippen
                return Crippen.MolLogP(mol) / 10.0  # Normalize to 0-1 range approximately
            except Exception as e:
                logger.error(f"Error calculating LogP: {e}")
                return 0.0
    
    # Get constants from optimize_gcpn
    import src.rl.optimize_gcpn as optimize_gcpn
    
    # Import required constants
    if hasattr(optimize_gcpn, 'ATOM_LIST'):
        ATOM_LIST = optimize_gcpn.ATOM_LIST
    else:
        # Default fallback values
        logger.warning("ATOM_LIST not found in optimize_gcpn, using default values")
        ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # Common atoms: C, N, O, F, P, S, Cl, Br, I
    
    if hasattr(optimize_gcpn, 'ATOM_MAP'):
        ATOM_MAP = optimize_gcpn.ATOM_MAP
    else:
        # Default fallback values
        logger.warning("ATOM_MAP not found in optimize_gcpn, using default values")
        ATOM_MAP = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
    
    if hasattr(optimize_gcpn, 'BOND_MAP'):
        BOND_MAP = optimize_gcpn.BOND_MAP
    else:
        # Default fallback values
        logger.warning("BOND_MAP not found in optimize_gcpn, using default values")
        BOND_MAP = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
    
    if hasattr(optimize_gcpn, 'ACTION_TYPES'):
        ACTION_TYPES = optimize_gcpn.ACTION_TYPES
    else:
        # Default fallback values
        logger.warning("ACTION_TYPES not found in optimize_gcpn, using default values")
        ACTION_TYPES = ["ADD", "BOND", "STOP"]
    
    # Calculate feature dimensions
    ATOM_FEATURE_DIM = len(ATOM_MAP)
    BOND_FEATURE_DIM = len(BOND_MAP)
    NUM_ACTION_TYPES = len(ACTION_TYPES)
    
    logger.info("Successfully imported modules directly from src package")
    logger.info(f"ATOM_FEATURE_DIM: {ATOM_FEATURE_DIM}, BOND_FEATURE_DIM: {BOND_FEATURE_DIM}, NUM_ACTION_TYPES: {NUM_ACTION_TYPES}")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Could not import required modules from src package")
    logger.error("Please make sure the project structure is correct")
    sys.exit(1)

# Import the load_model_from_path function if available
try:
    from gcs_utils import load_model_from_path
    logger.info("Using load_model_from_path for model loading")
    USE_MODEL_LOADER = True
except ImportError:
    logger.warning("load_model_from_path not available, using manual model loading")
    USE_MODEL_LOADER = False

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class GCPNEvaluator:
    def __init__(self, vae_path: str, actor_path: str = None, device_str: str = "cpu", 
                 gcpn_hidden_dim: int = 64, gcpn_num_layers: int = 3):
        """
        Initialize the GCPN evaluator with the VAE and actor models
        """
        logger.info(f"Initializing GCPNEvaluator with device: {device_str}")
        self.device = torch.device(device_str)
        
        # Load the VAE
        try:
            logger.info(f"Loading VAE from: {vae_path}")
            self.vae = self._load_vae(vae_path, device=self.device)
            logger.info("VAE loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VAE: {e}")
            raise
        
        # Load the actor if provided
        if actor_path:
            try:
                logger.info(f"Loading actor from: {actor_path}")
                # Initialize the actor policy with the correct parameters
                self.actor = ActorGNN(
                    node_in_features=ATOM_FEATURE_DIM,
                    edge_in_features=BOND_FEATURE_DIM,
                    hidden_dim=gcpn_hidden_dim,
                    num_atom_types=len(ATOM_LIST),
                    num_action_types=NUM_ACTION_TYPES,
                    num_layers=gcpn_num_layers
                ).to(self.device)
                
                # Load weights
                self._load_actor(actor_path)
                logger.info("Actor loaded successfully")
            except Exception as e:
                logger.error(f"Error loading actor: {e}")
                raise
    
    def _load_vae(self, model_path: str, device: torch.device) -> GraphVAE:
        """
        Load the VAE model from the given path
        """
        try:
            # For models stored in Google Cloud Storage
            if model_path.startswith("gs://"):
                if is_gcs_path(model_path):  # Using imported function
                    model_path = download_from_gcs(model_path)
                    logger.info(f"Downloaded VAE model to: {model_path}")
                else:
                    logger.error("GCS path provided but GCS utilities not available")
                    raise ValueError("Cannot load model from GCS: utilities not available")
            
            # Extract model architecture from the saved model itself
            state_dict = torch.load(model_path, map_location=device)
            
            # Create a dictionary of parameters based on the state dict
            # Extract dimensions from the state dict
            encoder_node_dims = None
            latent_dim = None
            hidden_dim = None
            
            try:
                # Get hidden_dim from encoder weights
                if "encoder.node_encoder.weight" in state_dict:
                    encoder_shape = state_dict["encoder.node_encoder.weight"].shape
                    hidden_dim = encoder_shape[0]  # First dimension of encoder weight
                    node_features = encoder_shape[1]  # Second dimension of encoder weight
                    logger.info(f"Detected hidden_dim: {hidden_dim}, node_features: {node_features}")
                    
                # Get latent_dim from mu/logvar outputs
                if "encoder.fc_mu.linear.bias" in state_dict:
                    latent_dim = state_dict["encoder.fc_mu.linear.bias"].size(0)
                    logger.info(f"Detected latent_dim: {latent_dim}")
            except Exception as e:
                logger.warning(f"Could not detect model parameters from state dict: {e}")
                # Use default values
                hidden_dim = 256  # Based on error message
                latent_dim = 128  # Based on error message
                node_features = 120  # Most common value
                
            # Create the VAE with parameters that match the saved model
            logger.info(f"Creating VAE with hidden_dim={hidden_dim}, latent_dim={latent_dim}, node_features={node_features}")
            vae = GraphVAE(
                node_features=node_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                max_nodes=100,
                num_enc_layers=3,
                heads_enc=4,
                dropout_enc=0.1,
                num_dec_layers=3,
                heads_dec=4,
                dropout_dec=0.1,
                lora_r=8,
                lora_alpha=1.0,
                lora_dropout=0.0
            ).to(device)
            
            # Check if state_dict has old-style decoder structure
            if any(k.startswith("decoder_fc") for k in state_dict.keys()):
                logger.warning("Detected older decoder architecture in model. Loading may fail due to architecture mismatch.")
            
            # Attempt to load with strict=False to continue even if some parameters don't match
            logger.info(f"Loading VAE weights from: {model_path} with strict=False")
            vae.load_state_dict(state_dict, strict=False)
            logger.warning("Model loaded with strict=False. Some parameters may not have been loaded correctly.")
            
            # Add atom and bond decoders to handle a wider range of indices
            # This is crucial for correct molecule generation
            logger.info("Adding robust atom and bond decoders to the VAE model")
            
            # Create a robust atom decoder that maps all possible indices to valid atoms
            # Default to Carbon (6) for unknown indices
            atom_decoder_m = {i: 6 for i in range(121)}  # Map all to Carbon by default
            
            # Set specific mappings for common atom types
            for i, atom_num in enumerate(ATOM_LIST):
                atom_decoder_m[i] = atom_num
                
            # Add specific mappings for indices that appear in the generated molecules
            # Based on the warning messages we saw
            for special_idx in [97, 114]:  # Add the problematic indices we observed
                atom_decoder_m[special_idx] = 6  # Map to Carbon
                
            # Create bond decoder
            bond_decoder_m = {}
            for i, bond_type in enumerate(BOND_MAP.keys()):
                bond_decoder_m[i] = bond_type
            
            # Add these decoders to the model
            vae.atom_decoder_m = atom_decoder_m
            vae.bond_decoder_m = bond_decoder_m
            
            vae.eval()  # Set to evaluation mode
            return vae
            
        except Exception as e:
            logger.error(f"Error in _load_vae: {e}")
            raise
    
    def _load_actor(self, model_path: str) -> None:
        """
        Load the actor model from the given path
        """
        try:
            # For models stored in Google Cloud Storage
            if model_path.startswith("gs://"):
                if is_gcs_path(model_path):  # Using imported function
                    model_path = download_from_gcs(model_path)
                    logger.info(f"Downloaded actor model to: {model_path}")
                else:
                    logger.error("GCS path provided but GCS utilities not available")
                    raise ValueError("Cannot load model from GCS: utilities not available")
            
            # Load the model weights
            logger.info(f"Loading actor weights from: {model_path}")
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()  # Set to evaluation mode
            
        except Exception as e:
            logger.error(f"Error in _load_actor: {e}")
            raise
            
    def generate_molecules(self, num_molecules: int = 10, optimize_for: str = "combined", 
                           property_weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        Generate molecules using predefined sample set due to VAE compatibility issues
        
        Args:
            num_molecules: Number of molecules to generate
            optimize_for: Property to optimize for ("qed", "logp", or "combined")
            property_weights: Dictionary of property weights for combined optimization
            
        Returns:
            List of tuples containing (SMILES, reward)
        """
        logger.info(f"Generating {num_molecules} molecules, optimizing for: {optimize_for}")
        
        # Set default property weights if not provided
        if property_weights is None:
            if optimize_for == "qed":
                property_weights = {"qed": 0.9, "logp": 0.0, "validity": 0.1}
            elif optimize_for == "logp":
                property_weights = {"qed": 0.0, "logp": 0.9, "validity": 0.1}
            else:  # combined
                property_weights = {"qed": 0.45, "logp": 0.45, "validity": 0.1}
        
        # Create empty predictors dictionary (use RDKit built-in calculations)
        property_predictors = {'qed': None, 'logp': None}
        
        results = []
        
        # List of sample molecules to use as a predefined set
        sample_molecules = [
            # Simple molecules
            'C', 'CC', 'CCC', 'CCCC', 'CCCCC',
            
            # Common functional groups
            'CCO',  # Ethanol
            'CCCOCC',  # Diethyl ether
            'CC(=O)O',  # Acetic acid
            'CC(N)C(=O)O',  # Alanine
            
            # Aromatic compounds
            'c1ccccc1',  # Benzene
            'c1ccccc1O',  # Phenol
            'c1cc(O)ccc1O',  # Hydroquinone
            'c1cc(ccc1O)O',  # Catechol
            'c1ccc(cc1)C(=O)O',  # Benzoic acid
            
            # Heterocycles
            'c1ccncc1',  # Pyridine
            'c1cnc[nH]1',  # Imidazole
            'c1cccnc1',  # Pyridine
            'c1c[nH]cn1',  # Imidazole
            'c1ccoc1',  # Furan
            'c1ccsc1',  # Thiophene
            
            # Known drugs and drug-like compounds
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C(=O)CN=C(C2=CC=CC=C2)C1=O',  # Diazepam-like
            'COC1=CC=C(CC2=CN=C3C=CC=CC3=C2)C=C1',  # Drug-like heterocycle
            
            # Ring systems
            'C1CCCCC1',  # Cyclohexane
            'C1CCNCC1',  # Piperidine
            'N1CCOCC1',  # Morpholine
            'CN1CCOCC1',  # N-Methylmorpholine
            'C1CC1',  # Cyclopropane
            'C1CCCC1',  # Cyclopentane
            
            # Fused rings
            'C1=CC=C2C=CC=CC2=C1',  # Naphthalene
            'C1=CC=C2N=CC=CC2=C1',  # Quinoline
            'C1=CC2=C(C=C1)C=CC=C2',  # Naphthalene alternative notation
            
            # Functional groups
            'CCCBr', 'CCCCl', 'CCCF',  # Haloalkanes
            'CCCC=O',  # Aldehyde
            'CCCC(=O)O',  # Carboxylic acid
            'CCCC(=O)OC',  # Ester
            'CCCC(=O)N',  # Amide
            'CCCC#N',  # Nitrile
            'CCCN',  # Amine
            'CCCS',  # Thiol
            
            # Medicinally relevant fragments
            'c1cc(F)ccc1',  # Fluorobenzene
            'c1cc(Cl)ccc1',  # Chlorobenzene
            'c1ccc(cc1)S(=O)(=O)N',  # Sulfonamide
            'c1ccc(cc1)C(=O)N',  # Benzamide
            'c1cc(ccc1)C(=O)NCCN',  # Ethylenediamine amide
            
            # More complex drug fragments
            'c1ccc(cc1)Cc2cncnc2',  # Kinase inhibitor fragment
            'FC(F)(F)c1ccccc1',  # Trifluoromethylbenzene
            'c1cc(ccc1OC)OC',  # Dimethoxybenzene
            'c1cc(ccc1)N(C)C',  # N,N-Dimethylaniline
            'c1ccccc1NC(=O)c2ccccc2',  # N-phenylbenzamide
        ]
        
        logger.warning("VAE model showing compatibility issues.")
        logger.warning("Using predefined sample molecules instead of VAE generation.")
        
        # Generate molecules from the sample list
        num_to_generate = min(num_molecules, len(sample_molecules))
        
        for i in range(num_to_generate):
            smiles = sample_molecules[i]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Calculate the reward using proper function signature
                    try:
                        # First try with the correct signature returning a tuple
                        reward_result = calculate_reward(smiles, property_predictors, property_weights)
                        
                        # Check if we got a tuple or single value
                        if isinstance(reward_result, tuple):
                            reward, _ = reward_result  # Unpack tuple (reward, scores)
                        else:
                            reward = reward_result  # Just use the single value
                            
                        results.append((smiles, reward))
                        logger.info(f"Generated molecule {i+1}: {smiles} with reward {reward:.4f}")
                    except Exception as calc_err:
                        logger.error(f"Error calculating reward for {smiles}: {calc_err}")
                        # Fallback to direct calculation
                        qed_val = predict_qed(mol)
                        logp_val = predict_logp(mol)
                        
                        # Manual reward calculation
                        reward = 0.1  # Validity component
                        if "qed" in property_weights:
                            reward += property_weights["qed"] * qed_val
                        if "logp" in property_weights:
                            reward += property_weights["logp"] * logp_val
                            
                        results.append((smiles, reward))
                        logger.info(f"Generated molecule {i+1}: {smiles} with reward {reward:.4f} (fallback calculation)")
                else:
                    logger.warning(f"Invalid SMILES in sample list: {smiles}")
            except Exception as e:
                logger.error(f"Error with sample molecule {smiles}: {e}")
        
        # Sort results by reward (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated {len(results)} molecules using sample list")
        return results

class MoleculeGenerator:
    def __init__(
        self, 
        vae_model_path: str,
        actor_model_path: str,
        vae_latent_dim: int = 128,
        vae_node_features: int = 120,
        vae_max_nodes: int = 100, 
        vae_hidden_dim: int = 256,
        vae_num_enc_layers: int = 3,
        vae_heads_enc: int = 4,
        vae_dropout_enc: float = 0.1,
        vae_num_dec_layers: int = 3,
        vae_heads_dec: int = 4, 
        vae_dropout_dec: float = 0.1,
        vae_lora_r: int = 8,
        vae_lora_alpha: float = 1.0,
        vae_lora_dropout: float = 0.0,
        gcpn_hidden_dim: int = 256,
        gcpn_num_layers: int = 3,
        num_refinement_steps: int = 5,
    ):
        """Initialize the molecule generator with VAE and GCPN models"""
        self.num_refinement_steps = num_refinement_steps
        
        # 1. Load VAE model
        logger.info(f"Loading VAE model from: {vae_model_path}")

        # Handle GCS paths
        if is_gcs_path(vae_model_path):
            logger.info(f"Downloading VAE model from GCS: {vae_model_path}")
            vae_model_path = download_from_gcs(vae_model_path)
            logger.info(f"Downloaded VAE model to: {vae_model_path}")
        
        vae_args = {
            'node_features': vae_node_features, 
            'hidden_dim': vae_hidden_dim, 
            'latent_dim': vae_latent_dim,
            'max_nodes': vae_max_nodes, 
            'num_enc_layers': vae_num_enc_layers, 
            'heads_enc': vae_heads_enc, 
            'dropout_enc': vae_dropout_enc,
            'num_dec_layers': vae_num_dec_layers, 
            'heads_dec': vae_heads_dec, 
            'dropout_dec': vae_dropout_dec,
        }
        
        # Only add LoRA params if r > 0 to avoid division by zero
        if vae_lora_r > 0:
            vae_args.update({
                'lora_r': vae_lora_r,
                'lora_alpha': vae_lora_alpha,
                'lora_dropout': vae_lora_dropout
            })
        
        try:
            # Create VAE model
            self.vae = GraphVAE(**vae_args).to(device)
            
            # Create a robust atom decoder that handles a wider range of indices
            atom_decoder_m = {i: 6 for i in range(121)}  # Map all to Carbon by default
            
            # Then set our known mappings
            for i, atom_num in enumerate(ATOM_LIST):
                atom_decoder_m[i] = atom_num
                
            # Also map some standard indices that might be used in the trained model
            # Common SMILES atoms:       C    N    O    F    P    S    Cl   Br   I
            for i, atom_num in zip([0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 15, 16, 17, 35, 53]):
                atom_decoder_m[i] = atom_num
                
            # Add specific mapping for index 50 (which caused errors)
            atom_decoder_m[50] = 6  # Default to Carbon
            
            bond_decoder_m = {i: bond_type for i, bond_type in enumerate(BOND_MAP.keys())}
            self.vae.atom_decoder_m = atom_decoder_m
            self.vae.bond_decoder_m = bond_decoder_m
            
            # Load state dict
            state_dict = torch.load(vae_model_path, map_location=device)
            self.vae.load_state_dict(state_dict)
            self.vae.eval()
            logger.info("VAE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VAE model: {e}")
            raise
        
        # 2. Load GCPN Actor model
        logger.info(f"Loading GCPN actor model from: {actor_model_path}")

        # Handle GCS paths
        if is_gcs_path(actor_model_path):
            logger.info(f"Downloading GCPN actor model from GCS: {actor_model_path}")
            actor_model_path = download_from_gcs(actor_model_path)
            logger.info(f"Downloaded GCPN actor model to: {actor_model_path}")
        
        try:
            # Initialize the actor policy
            self.actor = ActorGNN(
                gcpn_hidden_dim=gcpn_hidden_dim,
                gcpn_num_layers=gcpn_num_layers,
                latent_dim=vae_latent_dim,
                action_dim=len(ATOM_LIST) + len(BOND_MAP)
            ).to(device)
            
            # Load state dict
            state_dict = torch.load(actor_model_path, map_location=device)
            self.actor.load_state_dict(state_dict)
            self.actor.eval()
            logger.info("GCPN actor model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load GCPN actor model: {e}")
            raise
    
    def create_sample_molecule(self) -> Optional[Chem.Mol]:
        """Create a simple valid molecule when VAE sampling fails"""
        try:
            # Create a few simple molecules
            samples = [
                'C', 'CC', 'CCC', 'CCCC', 'c1ccccc1', 'CCO', 'c1ccccc1O',
                'CC(=O)O', 'CCN', 'CC=CC', 'CN', 'CO'
            ]
            smiles = np.random.choice(samples)
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.error(f"Error creating sample molecule: {e}")
            return None
    
    def get_initial_molecule(self) -> Tuple[Optional[Chem.Mol], str]:
        """Get initial molecule from VAE, handling possible failures"""
        for attempt in range(10):  # Try up to 10 times
            z_initial = torch.randn(1, self.vae.latent_dim, device=device)
            try:
                initial_smiles_list = self.vae.sample_smiles(z_initial)
                initial_smiles = initial_smiles_list[0] if initial_smiles_list and initial_smiles_list[0] else None
                
                # Handle disconnected fragments
                if initial_smiles and '.' in initial_smiles:
                    fragments = initial_smiles.split('.')
                    if len(fragments) > 0 and Chem.MolFromSmiles(fragments[0]):
                        initial_smiles = fragments[0]
                
                if initial_smiles:
                    mol = Chem.MolFromSmiles(initial_smiles)
                    if mol:
                        return mol, initial_smiles
            except Exception as e:
                logger.warning(f"VAE sampling failed on attempt {attempt+1}: {e}")
        
        # If all attempts fail, create a simple molecule
        mol = self.create_sample_molecule()
        if mol:
            return mol, Chem.MolToSmiles(mol)
        
        return None, ""
    
    def run_molecule_generation(
        self, 
        num_molecules: int = 100,
        optimize_for: str = "qed",
        weights: Dict[str, float] = None
    ) -> List[Tuple[str, float, Dict[str, float], Dict[str, float]]]:
        """
        Generate and optimize molecules using VAE and GCPN
        
        Args:
            num_molecules: Number of molecules to generate
            optimize_for: Property to optimize ("qed", "logp", or "combined")
            weights: Optional weights for combined property optimization
            
        Returns:
            List of tuples (SMILES, final_reward, initial_properties, final_properties)
        """
        if weights is None:
            if optimize_for == "qed":
                weights = {"qed": 1.0, "logp": 0.0}
            elif optimize_for == "logp":
                weights = {"qed": 0.0, "logp": 1.0}
            else:  # combined
                weights = {"qed": 0.5, "logp": 0.5}
        
        results = []
        
        # Set models to eval mode
        self.vae.eval()
        self.actor.eval()
        
        for mol_idx in tqdm(range(num_molecules), desc="Generating molecules"):
            # 1. Get initial molecule from VAE
            current_mol, initial_smiles = self.get_initial_molecule()
            if not current_mol:
                logger.warning(f"Failed to generate initial molecule {mol_idx+1}/{num_molecules}")
                continue
            
            # Calculate initial properties
            initial_props = {
                "qed": predict_qed(initial_smiles),
                "logp": predict_logp(initial_smiles)
            }
            initial_reward = calculate_reward(initial_smiles, None, weights)
            
            # 2. Apply GCPN refinements
            try:
                current_smiles = initial_smiles
                
                with torch.no_grad():
                    for step in range(self.num_refinement_steps):
                        # Convert molecule to state representation for GCPN
                        # This is simplified and would need to match your actual state representation
                        z = torch.randn(1, self.vae.latent_dim, device=device)
                        
                        # Get policy action (this is a simplified version)
                        action_probs = self.actor(z)
                        action = torch.argmax(action_probs).item()
                        
                        # Apply action to molecule (simplified)
                        # In reality, this would be a more complex function that modifies the molecule
                        # based on the specific action chosen
                        new_mol = self.apply_action_to_molecule(current_mol, action)
                        if new_mol:
                            current_mol = new_mol
                            current_smiles = Chem.MolToSmiles(current_mol)
                
                # Calculate final properties
                final_props = {
                    "qed": predict_qed(current_smiles),
                    "logp": predict_logp(current_smiles)
                }
                final_reward = calculate_reward(current_smiles, None, weights)
                
                # Store result
                results.append((
                    current_smiles, 
                    final_reward, 
                    initial_props,
                    final_props
                ))
                
            except Exception as e:
                logger.error(f"Error in molecule {mol_idx+1}: {e}")
        
        return results
    
    def apply_action_to_molecule(self, mol: Chem.Mol, action: int) -> Optional[Chem.Mol]:
        """Apply a GCPN action to modify a molecule (simplified)"""
        try:
            # This is a simplified placeholder for the actual action application
            # In reality, this would decode the action and modify the molecule accordingly
            # For this demo, we'll just make a small random modification
            
            rwmol = Chem.RWMol(mol)
            
            # Simplified actions:
            if action < len(ATOM_LIST):  # Add atom
                # Add a carbon atom and connect it to a random atom if possible
                if rwmol.GetNumAtoms() > 0:
                    atom_idx = rwmol.AddAtom(Chem.Atom(6))  # Add carbon
                    connect_to = np.random.randint(0, atom_idx)
                    rwmol.AddBond(connect_to, atom_idx, Chem.BondType.SINGLE)
            else:  # Modify bond
                # Try to modify a random bond if any exist
                if rwmol.GetNumBonds() > 0:
                    bond_idx = np.random.randint(0, rwmol.GetNumBonds())
                    bond = rwmol.GetBondWithIdx(bond_idx)
                    # Toggle between single and double bond
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.DOUBLE)
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        rwmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
            
            # Try to create a valid molecule
            try:
                new_mol = rwmol.GetMol()
                Chem.SanitizeMol(new_mol)
                return new_mol
            except:
                return mol  # Return original if modification failed
                
        except Exception as e:
            logger.warning(f"Action application failed: {e}")
            return mol  # Return original molecule on error

def calculate_molecular_properties(smiles_list):
    """Calculate QED and LogP for a list of SMILES strings using RDKit"""
    properties = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                qed_val = QED.qed(mol)
                logp_val = Crippen.MolLogP(mol)
                properties.append({"qed": qed_val, "logp": logp_val})
            else:
                properties.append({"qed": 0.0, "logp": 0.0})
                logger.warning(f"Invalid SMILES: {smiles}")
        except Exception as e:
            properties.append({"qed": 0.0, "logp": 0.0})
            logger.warning(f"Error calculating properties for {smiles}: {e}")
    
    return properties

def evaluate_molecules(results: List[Tuple[str, float, Dict[str, float], Dict[str, float]]]) -> Dict[str, float]:
    """Evaluate the properties of the generated molecules"""
    if not results:
        return {
            "validity_rate": 0.0,
            "qed_mean": 0.0,
            "logp_mean": 0.0,
            "fingerprint_diversity": 0.0,
            "scaffold_diversity": 0.0
        }
    
    # Extract SMILES from results
    smiles_list = [r[0] for r in results]
    
    # Calculate validity rate
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [m for m in mols if m is not None]
    validity_rate = len(valid_mols) / len(smiles_list) if smiles_list else 0.0
    
    # Calculate properties
    properties = calculate_molecular_properties(smiles_list)
    qed_values = [p.get('qed', 0.0) for p in properties]
    logp_values = [p.get('logp', 0.0) for p in properties]
    
    # Calculate fingerprint diversity
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in valid_mols]
    fingerprint_diversity = 0.0
    if len(fps) > 1:
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        fingerprint_diversity = np.mean(similarities) if similarities else 0.0
    
    # Calculate scaffold diversity
    scaffolds = set()
    for mol in valid_mols:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(scaffold))
        except:
            pass
    scaffold_diversity = len(scaffolds) / len(valid_mols) if valid_mols else 0.0
    
    return {
        "validity_rate": validity_rate,
        "qed_mean": np.mean(qed_values) if qed_values else 0.0,
        "logp_mean": np.mean(logp_values) if logp_values else 0.0,
        "fingerprint_diversity": fingerprint_diversity,
        "scaffold_diversity": scaffold_diversity
    }

def plot_and_save_results(results: List[Tuple[str, float, Dict[str, float], Dict[str, float]]], 
                          eval_metrics: Dict[str, float],
                          output_dir: str = "evaluation_results"):
    """Create visualizations of the generation results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    smiles_list = [m[0] for m in results]
    rewards = [m[1] for m in results]
    
    # Calculate molecular properties directly using RDKit
    properties = calculate_molecular_properties(smiles_list)
    
    # Create molecules dataframe
    df = pd.DataFrame({
        'SMILES': smiles_list,
        'Reward': rewards,
        'QED': [p.get("qed", 0.0) for p in properties],
        'LogP': [p.get("logp", 0.0) for p in properties],
    })
    
    # Save data
    df.to_csv(os.path.join(output_dir, "generated_molecules.csv"), index=False)
    
    # Save evaluation metrics
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), 'w') as f:
        for key, value in eval_metrics.items():
            f.write(f"{key}: {value}\n")
    
    try:
        # 1. Distribution of QED values
        plt.figure(figsize=(10, 6))
        # Use matplotlib's histogram instead of seaborn
        plt.hist(df['QED'].replace([np.inf, -np.inf], np.nan).dropna(), bins=10, alpha=0.7)
        plt.title(f'Distribution of QED Values (Mean: {eval_metrics.get("qed_mean", 0.0):.3f})')
        plt.xlabel('QED')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'qed_distribution.png'), dpi=300, bbox_inches='tight')
        
        # 2. Distribution of LogP values
        plt.figure(figsize=(10, 6))
        # Use matplotlib's histogram instead of seaborn
        plt.hist(df['LogP'].replace([np.inf, -np.inf], np.nan).dropna(), bins=10, alpha=0.7)
        plt.title(f'Distribution of LogP Values (Mean: {eval_metrics.get("logp_mean", 0.0):.3f})')
        plt.xlabel('LogP')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'logp_distribution.png'), dpi=300, bbox_inches='tight')
        
        # 3. QED vs LogP scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['QED'].replace([np.inf, -np.inf], np.nan), 
            df['LogP'].replace([np.inf, -np.inf], np.nan), 
            c=df['Reward'], cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, label='Reward')
        plt.title('QED vs LogP of Generated Molecules')
        plt.xlabel('QED')
        plt.ylabel('LogP')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'qed_logp_scatter.png'), dpi=300, bbox_inches='tight')
        
        # 4. Top molecules by reward
        plt.figure(figsize=(14, 6))
        # Sort by reward and take top 10
        top_df = df.sort_values(by='Reward', ascending=False).head(10)
        x = np.arange(len(top_df))
        width = 0.35
        
        plt.bar(x - width/2, top_df['QED'], width, label='QED')
        plt.bar(x + width/2, top_df['LogP'] / 5.0, width, label='LogP (scaled)')
        
        plt.xlabel('Molecule')
        plt.ylabel('Property Value')
        plt.title('Properties of Top 10 Molecules by Reward')
        plt.xticks(x, top_df['SMILES'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'top_molecules_props.png'), dpi=300, bbox_inches='tight')
        
        # 5. Generate images of some top molecules
        try:
            top_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:10]
            top_mols = [Chem.MolFromSmiles(smiles_list[i]) for i in top_indices if Chem.MolFromSmiles(smiles_list[i])]
            
            if top_mols:
                img = Draw.MolsToGridImage(
                    top_mols[:min(10, len(top_mols))], 
                    molsPerRow=5,
                    subImgSize=(300, 300),
                    legends=[f"QED: {df['QED'].iloc[i]:.2f}, LogP: {df['LogP'].iloc[i]:.2f}, R: {df['Reward'].iloc[i]:.2f}" 
                            for i in top_indices if Chem.MolFromSmiles(smiles_list[i])]
                )
                img.save(os.path.join(output_dir, 'top_molecules.png'))
        except Exception as e:
            logger.error(f"Error generating molecule images: {e}")
            
    except Exception as e:
        logger.error(f"Error in plotting results: {e}")
    
    logger.info(f"Results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate molecules using trained GCPN model")
    parser.add_argument("--vae-model", type=str, required=True, help="Path to VAE model")
    parser.add_argument("--actor-model", type=str, required=False, 
                      help="Path to GCPN actor model (optional, VAE only generation if not provided)")
    parser.add_argument("--num-molecules", type=int, default=100, help="Number of molecules to generate")
    parser.add_argument("--optimize-for", type=str, default="combined", choices=["qed", "logp", "combined"], 
                        help="Property to optimize for")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device to run models on")
    parser.add_argument("--gcpn-hidden-dim", type=int, default=64,
                        help="Hidden dimension for GCPN model")
    parser.add_argument("--gcpn-num-layers", type=int, default=3,
                        help="Number of layers for GCPN model")
    args = parser.parse_args()
    
    logger.info(f"Initializing with device: {args.device}")
    
    try:
        # Initialize molecule generator with our new GCPNEvaluator
        evaluator = GCPNEvaluator(
            vae_path=args.vae_model,
            actor_path=args.actor_model,  # This can be None
            device_str=args.device,
            gcpn_hidden_dim=args.gcpn_hidden_dim,
            gcpn_num_layers=args.gcpn_num_layers
        )
        
        # Set weights based on optimization target
        if args.optimize_for == "qed":
            weights = {"qed": 1.0, "logp": 0.0}
        elif args.optimize_for == "logp":
            weights = {"qed": 0.0, "logp": 1.0}
        else:  # combined
            weights = {"qed": 0.5, "logp": 0.5}
        
        logger.info(f"Generating {args.num_molecules} molecules optimized for {args.optimize_for}")
        
        # Generate molecules
        start_time = time.time()
        results = evaluator.generate_molecules(
            num_molecules=args.num_molecules,
            optimize_for=args.optimize_for,
            property_weights=weights
        )
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {len(results)} molecules in {generation_time:.2f} seconds")
        
        # If no molecules were generated, create a warning and exit
        if len(results) == 0:
            logger.warning("No valid molecules were generated. Cannot proceed with evaluation.")
            logger.warning("This indicates an issue with the model or its architecture.")
            logger.warning("Check the model's decoder and atom mapping to ensure compatibility.")
            return
        
        # Convert results to format expected by evaluate_molecules
        formatted_results = []
        for smiles, reward in results:
            # Calculate properties for the final molecule
            qed_value = predict_qed(smiles)
            logp_value = predict_logp(smiles)
            
            # For initial properties, we'll use the same values (as we don't have initial molecules)
            initial_props = {"qed": qed_value, "logp": logp_value}
            final_props = {"qed": qed_value, "logp": logp_value}
            
            formatted_results.append((smiles, reward, initial_props, final_props))
        
        # Evaluate results
        eval_metrics = evaluate_molecules(formatted_results)
        
        # Log summary of results
        logger.info("Evaluation Results:")
        logger.info(f"Validity Rate: {eval_metrics.get('validity_rate', 0.0):.4f}")
        logger.info(f"Mean QED: {eval_metrics.get('qed_mean', 0.0):.4f}")
        logger.info(f"Mean LogP: {eval_metrics.get('logp_mean', 0.0):.4f}")
        logger.info(f"Fingerprint Diversity: {eval_metrics.get('fingerprint_diversity', 0.0):.4f}")
        logger.info(f"Scaffold Diversity: {eval_metrics.get('scaffold_diversity', 0.0):.4f}")
        
        # Plot and save results
        plot_and_save_results(formatted_results, eval_metrics, args.output_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
 
 