# src/trait_engine.py
import joblib, numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "config" / "trait_model.pkl"

class TraitEngine:
    """Sample immutable trait rows for synthetic agents.

    Uses a fitted Gaussian copula to generate unlimited synthetic populations
    that preserve the correlation structure of the original 280 participants
    while only including traits actually needed by decision modules.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Expected fitted copula model at {MODEL_PATH}. Run 'python scripts/train_copula.py' first.")
            
        # Load the fitted copula model
        blob        = joblib.load(MODEL_PATH)
        self.Sigma  = blob["Sigma"]
        self.traits = blob["traits"]
        self.chol   = np.linalg.cholesky(self.Sigma)
        
        # Reconstruct decoder functions from stored data
        self.decode = {}
        for col, decoder_data in blob["decoders"].items():
            values = decoder_data['values']
            cdf = decoder_data['cdf']
            dtype = decoder_data['dtype']
            
            def make_decoder(vals, cdf_vals):
                def decoder(u):
                    return vals[np.searchsorted(cdf_vals, u, side="right")]
                return np.vectorize(decoder, otypes=[dtype])
            
            self.decode[col] = make_decoder(values, cdf)
        
        print(f"Loaded copula model with {len(self.traits)} traits")

    def sample(self, n_agents: int, seed: int):
        """Return a DataFrame with *n_agents* rows.

        Generates novel synthetic agents by:
        1. Drawing n i.i.d. standard-normal vectors
        2. Multiplying by Cholesky factor to impose correlations
        3. Converting to uniforms via normal CDF
        4. Decoding each uniform back to original trait scale
        """
        rng  = np.random.default_rng(seed)
        z    = rng.standard_normal((n_agents, len(self.traits))) @ self.chol.T
        u    = norm.cdf(z)
        cols = {name: self.decode[name](u[:, j])
                for j, name in enumerate(self.traits)}
        return pd.DataFrame(cols)
    
    def get_available_traits(self):
        """Return list of trait column names from fitted model."""
        return list(self.traits)