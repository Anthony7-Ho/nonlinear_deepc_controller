This directory is a bootstrap placeholder for the installed kernel DeePC bundle.

Why it exists:
- fresh clones need to build the package before any training data or predictors exist
- the real generated bundle lives in `data_processing/kernel_deepc_bundle/`
- `CMakeLists.txt` installs that real bundle when it exists locally, otherwise it installs this placeholder into `share/nonlinear_deepc_controller/data/kernel_deepc_bundle`

How to replace it:
1. Collect training and validation data with the joint impedance controller.
2. Run `data_processing/1joint_predictor.ipynb` to generate `data_processing/kernel_deepc_bundle/`.
3. Rebuild the package so the install tree picks up the real bundle.
