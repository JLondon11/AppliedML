# Chapter 5 — Accelerated MRI real-data reconstruction

**Master QA: BLOCKED for the named trained model.** `evaluate_fastmri.py` supports actual **raw single- or multi-coil fastMRI HDF5** data and produces numerical comparisons and five-panel scientific figures from code execution. It compares zero-filled reconstruction with an image-domain TV-regularized, acquired-k-space-data-consistent projection, **not** Adaptive-CS-Net, VarNet, or another trained model.

- Dataset source: [fastMRI](https://fastmri.med.nyu.edu/) / [NYU CAI2R](https://cai2r.net/resources/fastmri-dataset/), Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI* (2018).
- No real raw fastMRI file was present in this execution environment. The local tests use **generated k-space fixtures solely to check the implementation** and are not experimental evidence.
- Reproduce after independently obtaining the original, access-controlled raw HDF5 file:
```bash
pip install h5py numpy matplotlib scikit-image
python test_local.py
python evaluate_fastmri.py --h5 /original/fastMRI/file.h5 --slice-index 15 \
   --acceleration 4 --center-fraction 0.08 --output results/
```
- Outputs: `reconstruction_metrics.csv`, `real_fastmri_reconstruction.{png,pdf,svg}`, `experiment.json`. The file hash, slice and true sampling fraction are recorded. Reference is root-sum-squares of full-resolution IFFT coils; a Cartesian mask is retrospectively applied to full measurements, **not a clinically shortened acquisition**.
- The appropriate next step is paired trained-model reconstruction with original weights, a documented fastMRI validation split, matching Fourier conventions/mask and SSIM/PSNR, and a comparison against published results under identical evaluation. No patient image or measured model result is claimed here.
