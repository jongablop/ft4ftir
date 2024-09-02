from brukeropusreader import read_file
import numpy as np

def load_opus_file(opus_file_path):
    """
    Load an OPUS file and extract the relevant information.

    Parameters:
    - opus_file_path (str): Path to the OPUS file.

    Returns:
    - dict: A dictionary containing the interferogram, Fourier transform parameters, and signal parameters.
    """
    opus_file = read_file(opus_file_path)
    
    # Extract signal gain first (required in the interferogram processing)
    signal_gain = opus_file.get('Instrument').get('ASG')

    # Interferogram parameters
    interferogram_scaling_factor = opus_file.get('IgSm Data Parameter').get('CSF')
    laser_wavenumber = opus_file.get('Instrument').get('HFL')
    original_interferogram = opus_file['IgSm']
    x_index = opus_file.get_range('IgSm')
    
    scaled_interferogram = original_interferogram * interferogram_scaling_factor
    
    interferogram_parameters = {
        'x_index': np.array(x_index),
        'interferogram': np.array(scaled_interferogram),
        'laser_wavenumber': laser_wavenumber,
    }

    # Fourier transform parameters
    apodization_functions_naming = {
        'B3': 'BlackmanHarris3Term'
        # Add other mappings as needed
    }

    apodization_function = opus_file.get('Fourier Transformation').get('APF')

    ft_parameters = {
        'apodization_function_name': apodization_functions_naming.get(apodization_function, 'Unknown')
    }

    # Signal parameters
    signal_parameters = {
        'signal_gain': signal_gain
    }

    return {
        'interferogram': interferogram_parameters,
        'fourier_transform': ft_parameters,
        'signal': signal_parameters
    }
