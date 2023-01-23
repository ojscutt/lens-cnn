# lens-cnn
## Code used in paper Characterising Dark Matter Substructure in Gravitational Lens Galaxies with Deep Learning, DOI:[10.1137/22S1478033](https://doi.org/10.1137/22S1478033)

Files for the generation of simulated strong gravitational lensing images (with fixed or varied lens model parameters) populated by dark matter substructures with mass defined by a mass distribution power law variable. These images can be used train and test two convolutional neural networks in series - the first predicting the number of substructures, and the second using these number predictions to predict the power law slope of the distribution of substructure masses. 

### Notes:
- I've uploaded these as .py files, but I'd recommend opening into jupyter and splitting sections (marked by # ====...) into different cells and running separately to isolate any issues you might have (especially useful for the [n_cnn.py](n_cnn.py) and [a_cnn.py](a_cnn.py) files when testing)
- All .py files should be placed in the same directory - they should automatically detect and create required folders (dataset1, dataset2, models, logs) within this directory and populate these with images, logs, models etc

### Directions:
1. Training and testing set generation with [img_gen.py](img_gen.py) or [img_gen(vary).py](img_gen(vary).py)
   - Open [img_gen.py](img_gen.py) for images with fixed lens model parameters, or [img_gen(vary).py](img_gen(vary).py) for images with varied lens model parameters
   - Set variable 'set_size' to desired number of images in set
   - Set variable 'res_scale' to the resolution scaling factor, with 60x60 pixels being the base (60x60=1, 30x30=0.5 etc.)
   - Set variable 'im_set' to define image set to create (1 = train, 2 = test)
   - Alter other variables (Einstein radius, orientation, redshifts etc.) if required
   - Run, checking appropriate folders (and pretty lensing images) are created

2. Load and pickle images for CNNs with [img_load.py](img_load.py)
   - Open [img_load.py](img_load.py)
   - Set variable 'im_set' to define image set to load (1 = train, 2 = test)
   - Set variable 'res_scale' to match that used in training and testing set generation (60x60=1, 30x30=0.5 etc.)
   - Run, checking appropriate pickle files are created

3. Train and test N-CNN on prediction of substruture number with [n_cnn.py](n_cnn.py)
   - Open [n_cnn.py](n_cnn.py)
   - Alter 'name' string if running multiple tests to avoid models and logs saving over eachother
   - Alter other variables (conv filters, dropout, epochs etc.) if required
   - If training and testing in one run, ensure you change 'model' string further down to match defined 'name' string
   - Run, checking defined xlim and ylim are appropriate (might need changing to make graph neater, depending on CNN performance)

4. Train and test a-CNN on prediction of substruture mass distribution power law with [a_cnn.py](a_cnn.py)
   - Open [a_cnn.py](a_cnn.py)
   - Alter 'name' string if running multiple tests to avoid models and logs saving over eachother
   - Set variable 'res_scale' to match that used in training and testing set generation (60x60=1, 30x30=0.5 etc.)
   - Alter other variables (conv filters, dropout, epochs etc.) if required
   - If training and testing in one run, ensure you change 'model' string further down to match defined 'name' string
   - Run, checking defined xlim and ylim are appropriate (might need changing to make graph neater, depending on CNN performance)
