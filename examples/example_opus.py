import pandas as pd
from ft4ftir.interferogram import Interferogram
from ft4ftir.apodization import Apodization
from ft4ftir.fourier_transform import FourierTransform
from ft4ftir.io.bruker_opus import load_opus_file

# This example shows how to replicate the preprocessing and the Fourier
# transform of an interferogram stored in an OPUS file.

# Extract the parameters from the OPUS file
opus_file_path = './example_opus.0'
data = load_opus_file(opus_file_path)

interferogram_data = data.get('interferogram')

x_index = interferogram_data['x_index']
interferogram = interferogram_data['interferogram']
laser_wavenumber = interferogram_data['laser_wavenumber']

# right now we can only handle single interferograms
flux_size = int(interferogram.size / 2)
forward_x_index = x_index[:flux_size]
forward_interferogram = interferogram[:flux_size]

interferogram = Interferogram(
    x_index = forward_x_index,
    interferogram = forward_interferogram,
    laser_wavenumber = laser_wavenumber
)

ft_data = data.get('fourier_transform')
apodization_function_name = ft_data.get('apodization_function_name')

apodization = Apodization(
    name = apodization_function_name,
    size = forward_x_index.size,
)

fourier_transform = FourierTransform(
    interferogram = interferogram,
    apodization = apodization
)

flux, wavenumbers = fourier_transform()

signal_data = data.get('signal')
signal_gain = signal_data['signal_gain']

# Scale the flux by the signal gain
scaled_flux = flux / signal_gain

# Save to CSV
csv_filename = opus_file_path.split('.')[0] + '_forward.csv'
df = pd.DataFrame({
    'wavenumber / cm^-1': wavenumbers,
    'intensity / a.u.': scaled_flux
})
df.to_csv(csv_filename, index=False)
