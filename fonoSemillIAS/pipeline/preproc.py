from tqdm import tqdm

from fonoSemillIAS.preproc.othersFilters import DWT_filter
from fonoSemillIAS.preproc.othersFilters import nrp_filter
from fonoSemillIAS.preproc.temporalFilters import median_filter

def apply_preproc(signal, fs, torch):
  """
  Applies preprocessing to a signal.

  Parameters:
  - signal (array): The input signal.
  - fs (float): The sampling frequency of the signal.
  - torch (bool): Indicates whether to use PyTorch's adaptive threshold for the filter.

  Returns:
  - signal_pb (array): The processed signal after applying preprocessing.

  Raises:
  - Exception: If an error occurs during preprocessing.
  """

  try:
    progress_bar = tqdm(total=2, desc='Progress', position=0)
    # 1. DWT filter
    tqdm.write("Start step 1: DWT filter")
    signal_dwt, _ = DWT_filter(signal, wavelet="sym10", level=5, threshold_value=0.02)
    # Update progress bar
    progress_bar.update(1)

    # 2. Applying non-redundant noise reduction filter (NRP) to the DWT signal
    tqdm.write("Start step 2: Applying NRP filter")
    if torch:
        signal_pb = nrp_filter(signal=signal_dwt, std_tresh=4.5, fs=fs, torch=True)
    else:
        signal_pb = nrp_filter(signal=signal_dwt, std_tresh=0.5, fs=fs, torch=False)
    # Update progress bar
    progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
  except Exception as e:
    print(e)
    raise Exception("Error in preprocessing")
  
  return signal_pb