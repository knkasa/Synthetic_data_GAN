# https://docs.synthetic.ydata.ai/2.0/getting-started/installation/
# https://github.com/ydataai/ydata-synthetic/tree/dev/examples
# https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/DoppelGANger_FCC_MBA_Dataset.ipynb
# https://github.com/stakahashy/fingan

import pandas as pd
import matplotlib.pyplot as plt
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import os

os.chdir('C:/Users/knkas/Desktop/NLP_example')
mba_data = pd.read_csv("fcc_mba.csv")
numerical_cols = ["traffic_byte_counter", "ping_loss_rate"]
categorical_cols = [col for col in mba_data.columns if col not in numerical_cols]

# Defining model and training parameters
model_args = ModelParameters(batch_size=100,
                             lr=0.001,
                             betas=(0.2, 0.9),
                             latent_dim=20,
                             gp_lambda=2,
                             pac=1)

train_args = TrainParameters(epochs=10,  #400
                             sequence_length=56,
                             sample_length=8,
                             rounds=1,
                             measurement_cols=["traffic_byte_counter", "ping_loss_rate"])

# Training the DoppelGANger synthesizer
model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger',model_parameters=model_args)
model_dop_gan.fit(mba_data, train_args, num_cols=numerical_cols, cat_cols=categorical_cols)

# Generating new synthetic samples
synth_data = model_dop_gan.sample(n_samples=600)
synth_df = pd.concat(synth_data, axis=0)

print(synth_data)
