# QGP-PINN-model

# QGP-PINN: Physics-Informed Neural Network for QGP Thermodynamics

QGP-PINN is an end-to-end physics-informed machine learning pipeline designed to extract the thermodynamics of Quark-Gluon Plasma (QGP)[cite: 5]. Instead of predicting bulk thermodynamic quantities directly, the network learns a single latent variable: the effective coupling constant squared, $g^2(T,\mu_B)$[cite: 5]. 

---

## Installation & Requirements

The codebase requires Python 3.10 and several standard machine learning libraries[cite: 5]. Install the dependencies using the following command[cite: 5]:

pip install torch numpy pandas scikit-learn matplotlib

---

## File Structure

The repository is structured to separate model definition, training, evaluation, and plotting[cite: 5].

| File | Description |
| :--- | :--- |
| `qgp_pinn.py` | Defines the ResNet architecture (`QPMPinn`) that predicts $g^2(T,\mu_B)$[cite: 5]. |
| `train_final.py` | Trains the model on `Thermo_data_central.csv` and saves the checkpoint and loss history to `./model/`[cite: 5]. |
| `plot_results_final.py` | Loads the trained checkpoint, sweeps over T and $\mu_B/T$, and produces equation-of-state figures[cite: 5]. |
| `Test_Pipeline_Final.py` | Evaluates the trained checkpoint on the held-out 20% test split[cite: 5]. |

*(Note: Jupyter Notebook equivalents of these scripts, such as `train_final.ipynb` and `Test_Pipeline_Final.ipynb`, are also included for interactive step-by-step execution[cite: 2, 3])*

---

## How to Run

Execute the pipeline in the following standard order[cite: 5]:

1. **Train the model:** Run `train_final.ipynb` to train the ResNet and generate the `pinn_final_central.pth` checkpoint[cite: 5].
2. **Generate plots:** Run `plot_results_final.ipynb` to create the thermodynamic sweeps and save the results as a CSV[cite: 5].
3. **Evaluate performance:** Run `Test_Pipeline_Final.ipynb` to compute TrueMAPE, GRE, $R^2$, and Bias metrics on the unseen test set[cite: 5].

---

## Architecture & Physics Integration

The model embeds non-learnable physics equations directly into the forward pass so gradients flow backward through the entire physics chain[cite: 4, 5].

* **ML Backbone:** A 4-block Residual Network (ResNet) with SiLU activations takes normalized inputs $T/T_c$ and $\mu_B/T_c$[cite: 5].
* **Output Head:** A Softplus activation guarantees the predicted coupling $g^2$ remains strictly positive[cite: 5].
* **Mass Formulas:** Effective thermal masses for gluons ($m_g$), light quarks ($m_{ud}$), and strange quarks ($m_s$) are computed from the predicted $g^2$ using Quasiparticle Model (QPM) equations[cite: 5].
* **Partition Functions:** Log partition functions ($\ln Z$) are evaluated via a 50-point Gauss-Laguerre quadrature for both the quark and gluon sectors[cite: 5].
* **Thermodynamics:** Autograd derivatives of the partition function extract pressure ($P$), entropy density ($s$), baryon density ($n_B$), energy density ($\epsilon$), and the trace anomaly ($\Delta$)[cite: 5].

---

## Loss Function

The network is optimized using a 4-term Mean Absolute Error ($L_1$) loss against Wuppertal-Budapest Lattice QCD ground truth data[cite: 5]. 

* The target observables are $s$, $\Delta/T$, and $n_B$[cite: 5].
* An additional mass-consistency regularizer ($\mathcal{L}_{mass}$) softly anchors the mass ratios to the asymptotic-freedom limit when $T > 0.8 T_c$[cite: 5].

---

## Evaluation Metrics

The model is evaluated using a rigorous held-out test split of 20%[cite: 2, 5]. Performance metrics include TrueMAPE, Global Relative Error (GRE), $R^2$, and Relative Bias[cite: 2]. The average TrueMAPE across all five targets (P, $\epsilon$, s, $\Delta$, $n_B$) falls consistently between 0.51% and 0.63%, indicating a highly accurate fit[cite: 2, 5].