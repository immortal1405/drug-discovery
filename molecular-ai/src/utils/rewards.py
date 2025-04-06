# molecular-ai/src/utils/rewards.py

import logging
from typing import Tuple, Dict, Any, Optional

# Attempt to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not found. Reward calculation will be limited.")


# --- Property Prediction Functions ---
def predict_qed(mol: 'Chem.Mol', predictor_model: Optional[Any] = None) -> float:
    """
    Predicts or calculates QED for a molecule.
    Uses RDKit's built-in QED if no model provided.
    """
    if not RDKIT_AVAILABLE: return 0.0
    if predictor_model:
        # Implementation to use trained predictor model
        # Would need to convert mol to features expected by the model
        # Return predictor_model.predict(features)
        pass
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0 # Return low score if calculation fails

def predict_logp(mol: 'Chem.Mol', predictor_model: Optional[Any] = None) -> float:
    """
    Predicts or calculates LogP for a molecule.
    Uses RDKit's built-in Crippen LogP if no model provided.
    """
    if not RDKIT_AVAILABLE: return 0.0
    if predictor_model:
        # Implementation to use trained predictor model
        pass
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return -10.0 # Return low score if calculation fails

def predict_custom_property(mol: 'Chem.Mol', predictor_model: Any) -> float:
    """
    Placeholder for predicting a custom property using a specific model.
    """
    if not RDKIT_AVAILABLE or not predictor_model: return 0.0
    try:
        # Implementation to use custom model
        # Example: features = your_feature_extractor(mol)
        # prediction = predictor_model.predict(features)
        return 0.5 # Placeholder value
    except Exception:
        return 0.0

# --- Main Reward Calculation Function ---
def calculate_reward(
    smiles: str,
    property_predictors: Dict[str, Any],
    property_weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Calculates a composite reward for a given SMILES string based on validity
    and predicted properties.

    Args:
        smiles: The SMILES string of the molecule.
        property_predictors: Dictionary mapping property names to predictor models.
        property_weights: Optional dictionary mapping property names to weights.

    Returns:
        Tuple containing:
            - The total calculated reward (float).
            - A dictionary of individual property scores (Dict[str, Optional[float]]).
    """
    if not RDKIT_AVAILABLE:
        logging.warning("RDKit not available, returning 0 reward.")
        return 0.0, {}

    individual_scores = {}
    total_reward = 0.0

    # 1. Check Validity
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Invalid SMILES, return 0 reward
        individual_scores['validity'] = 0.0
        return 0.0, individual_scores
    else:
        # Assign a small reward just for validity
        individual_scores['validity'] = 1.0
        total_reward += 0.1  # Small base reward for being valid

    # 2. Calculate/Predict Properties
    property_fns = {
        'qed': predict_qed,
        'logp': predict_logp,
        # Add mappings for custom properties here
    }

    calculated_properties = {}
    for prop_name, predictor_model in property_predictors.items():
        if prop_name in property_fns:
            try:
                score = property_fns[prop_name](mol, predictor_model)
                calculated_properties[prop_name] = score
            except Exception as e:
                logging.warning(f"Failed to calculate property '{prop_name}' for {smiles}: {e}")
                calculated_properties[prop_name] = None
        elif predictor_model is not None: # Custom predictor
             try:
                score = predict_custom_property(mol, predictor_model)
                calculated_properties[prop_name] = score
             except Exception as e:
                logging.warning(f"Failed to calculate custom property '{prop_name}' for {smiles}: {e}")
                calculated_properties[prop_name] = None

    # 3. Combine scores into a single reward
    if not property_weights:
        # Default to equal weights
        property_weights = {}
        num_props = len(calculated_properties)
        if num_props > 0:
            for prop in calculated_properties:
                property_weights[prop] = 1.0 / num_props
        else:
            property_weights['validity'] = 1.0  # Just use validity if no predictors
    
    # Add validity weight if not specified
    if 'validity' not in property_weights:
        property_weights['validity'] = 0.1  # Small weight for validity

    # Calculate weighted sum of scores
    for prop_name, score in calculated_properties.items():
        individual_scores[prop_name] = score
        if score is not None:
            weight = property_weights.get(prop_name, 0.0)
            total_reward += weight * score

    # Add validity component
    total_reward += property_weights['validity'] * individual_scores['validity']

    return total_reward, individual_scores


if __name__ == '__main__':
    # Example usage
    if RDKIT_AVAILABLE:
        test_smiles = "CCO"  # Ethanol
        # Using built-in RDKit calculations
        predictors = {'qed': None, 'logp': None}
        weights = {'qed': 0.7, 'logp': 0.2, 'validity': 0.1}
        
        reward, scores = calculate_reward(test_smiles, predictors, weights)
        print(f"SMILES: {test_smiles}")
        print(f"Reward: {reward:.4f}")
        print(f"Scores: {scores}") 