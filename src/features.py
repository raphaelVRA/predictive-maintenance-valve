import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import correlate

def extract_features(pressure_segment, flow_segment):
    """
    Extrait les caractéristiques pertinentes des données de pression et de débit.
    
    Args:
        pressure_segment (array-like): Valeurs de pression pour un cycle (100 Hz)
        flow_segment (array-like): Valeurs de débit pour un cycle (10 Hz)
    
    Returns:
        dict: Dictionnaire des caractéristiques calculées:
            - skew_pressure: Asymétrie de la distribution des pressions
            - kurtosis_pressure: Aplatissement de la distribution des pressions
            - autocorr_pressure: Autocorrélation des pressions (lag=1)
            - derivative_max_pressure: Maximum de la dérivée de la pression
            - derivative_kurtosis_flow: Aplatissement de la dérivée du débit
            - integral_flow: Intégrale du débit sur le cycle
            - skew_flow: Asymétrie de la distribution du débit
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





