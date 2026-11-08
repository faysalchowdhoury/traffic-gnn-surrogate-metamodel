"""traffic-gnn-surrogate-metamodel.

Fast surrogates for a closed-loop signalised-corridor simulator: a message-passing graph neural
network (PyTorch Geometric) and gradient-boosted trees (LightGBM), trained on thousands of
simulated timing/demand scenarios, compared with linear and HCM-analytic baselines, versioned in a
file-based registry, monitored for input drift and served behind a FastAPI endpoint.
"""

__version__ = "1.0.0"
