import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import correlate

def extract_features(pressure_segment, flow_segment):
    """
    Extrait uniquement les caractéristiques sélectionnées des segments de pression (PS2) et de débit (FS1).
    :param pressure_segment: liste ou tableau numpy contenant les valeurs de pression pour un cycle
    :param flow_segment: liste ou tableau numpy contenant les valeurs de débit pour un cycle
    :return: dictionnaire de caractéristiques
    """
    pressure_segment = np.array(pressure_segment)
    flow_segment = np.array(flow_segment)

    features = {
        'skew_pressure': skew(pressure_segment),
        'kurtosis_pressure': kurtosis(pressure_segment),
        'autocorr_pressure': np.corrcoef(pressure_segment[:-1], pressure_segment[1:])[0, 1],
        'derivative_max_pressure': np.max(np.diff(pressure_segment)),

        'derivative_kurtosis_flow': kurtosis(np.diff(flow_segment)),
        'integral_flow': np.trapz(flow_segment),
        'skew_flow': skew(flow_segment),
    }

    return features





