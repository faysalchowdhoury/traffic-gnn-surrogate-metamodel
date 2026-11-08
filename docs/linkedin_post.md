The HCM delay equation explained less than half the variation in intersection delay across my corridor scenarios. A graph neural network explained 80%, and answered in 2 milliseconds.

I generated 1,500 timing and demand scenarios for a four-signal arterial and ran each through a closed-loop NEMA actuated-coordinated simulator (1.5 s per run). Then I trained four models to predict each intersection's delay and the corridor's fuel and CO2 without running the simulator.

Mean intersection-delay R² on 225 held-out scenarios:
🔹 HCM equation: 0.45
🔹 Ridge regression: 0.63
🔹 LightGBM: 0.77
🔹 GNN: 0.81 (MAE 4.8 s/veh)

The GNN treats intersections as nodes and links them with edges that carry what matters for progression: the travel time between signals and how their offsets line up with it. The HCM equation assumes random arrivals, so it can't tell a good offset from a bad one.

Three things I didn't expect:
1️⃣ The GNN was also the fastest learned model: 2.0 ms per scenario vs 8.6 ms for LightGBM. That's ~730× faster than simulating, so 1,000 timing alternatives screen in about 2 seconds.
2️⃣ For corridor fuel, plain ridge regression beat LightGBM. Fuel is nearly linear in volume, and trees approximate a straight line with steps.
3️⃣ All the models get worse in heavy congestion, where the training data is thin. A surrogate is for screening; the final alternatives still go back to simulation.

To catch the model being used outside its range, the API runs a drift test against the training data. On held-out scenarios 0 of 52 features flagged; with 30% more demand, 16 flagged.

Caveats: one corridor, one cycle length, a mesoscopic simulator rather than Vissim.

Repo: [link]

Signal timing folks: would a fast screening model like this change how many alternatives you evaluate?

#TrafficSignals #GraphNeuralNetworks #MachineLearning #MLOps #TrafficEngineering #SurrogateModeling #Python
