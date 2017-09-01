Here's how to run these.

First, you have to have python, opencv, numpy, and keras installed correctly.
The instructions to do so can be found here: https://github.com/layneson/fri-spring-2017/blob/master/README.md.

Once your environment is set up, you have to download and preprocess the IMFDB.
First download the `IMFDB_FIXED.zip` file from the shared drive (it's in the `Spring 2017` folder).
Unpack it in a dedicated folder so that you have the following structure:

    > DedicatedFolderName
        > IMFDB_FIXED
            > AamairKhan
            > Aarthi
            > AkshayKumar
            ...

Next, download the two preprocessing scripts:
- https://raw.githubusercontent.com/FRI-Deep-Learning/Transform-Images/master/LFW_img_transform.py
- https://raw.githubusercontent.com/FRI-Deep-Learning/Auto-Occlude/master/auto_occlude.py

Then run `python LFW_img_transform.py` and then `python auto_occlude.py` to preprocess the IMFDB.

Next, download the pickle_images and train_model scripts:
- https://raw.githubusercontent.com/FRI-Deep-Learning/initial-cnn/master/pickle_images.py
- https://raw.githubusercontent.com/FRI-Deep-Learning/initial-cnn/master/train_model.py

Run `python pickle_images.py` to generate training and testing splits. This only has to be done once... until we improve the pickle script.

Finally, run `python train_model.py`. This will train the model (slowly). When it is done, the trained model will be saved. Rename / move the model file (`finished_model.hdf5`) so that it doesn't get overwritten.

Changes can be made to the network in the `train_model.py` file. Once changes are made, the pickle script does NOT have to be run.
Simply invoke `python train_model.py`; that's enough (until we improve the pickle script).