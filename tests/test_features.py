import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import correlate


def extract_features(pressure_segment, flow_segment):
    """
    Extrait un ensemble complet de caractéristiques statistiques à partir des segments de pression (PS2) et de débit (FS1).
    :param pressure_segment: liste ou tableau numpy contenant les valeurs de pression pour un cycle
    :param flow_segment: liste ou tableau numpy contenant les valeurs de débit pour un cycle
    :return: dictionnaire de caractéristiques
    """
    pressure_segment = np.array(pressure_segment)
    flow_segment = np.array(flow_segment)

    # Caractéristiques sur la pression (PS2)
    features = {
        'mean_pressure': np.mean(pressure_segment),
        'max_pressure': np.max(pressure_segment),
        'min_pressure': np.min(pressure_segment),
        'std_pressure': np.std(pressure_segment),
        'median_pressure': np.median(pressure_segment),
        'skew_pressure': skew(pressure_segment),
        'kurtosis_pressure': kurtosis(pressure_segment),
        'ptp_pressure': np.ptp(pressure_segment),
        'mad_pressure': np.mean(np.abs(np.diff(pressure_segment))),
        'rise_time_pressure': np.argmax(pressure_segment),  # nb de points jusqu'au max
        'rms_pressure': np.sqrt(np.mean(pressure_segment ** 2)),  # Root Mean Square
        'zero_crossings_pressure': np.count_nonzero(np.diff(np.sign(pressure_segment))),  # Zero crossing count
        'slope_pressure': (pressure_segment[-1] - pressure_segment[0]) / len(pressure_segment),  # Pente de la pression
        'autocorr_pressure': np.corrcoef(pressure_segment[:-1], pressure_segment[1:])[0, 1],  # Autocorrélation
        'energy_pressure': np.sum(pressure_segment ** 2),  # Energie
        'derivative_max_pressure': np.max(np.diff(pressure_segment)),  # Maximum de la dérivée

        # Caractéristiques sur le débit (FS1)
        'mean_flow': np.mean(flow_segment),
        'max_flow': np.max(flow_segment),
        'min_flow': np.min(flow_segment),
        'std_flow': np.std(flow_segment),
        'median_flow': np.median(flow_segment),
        'skew_flow': skew(flow_segment),
        'kurtosis_flow': kurtosis(flow_segment),
        'ptp_flow': np.ptp(flow_segment),
        'mad_flow': np.mean(np.abs(np.diff(flow_segment))),
        'rms_flow': np.sqrt(np.mean(flow_segment ** 2)),  # Root Mean Square
        'zero_crossings_flow': np.count_nonzero(np.diff(np.sign(flow_segment))),  # Zero crossing count
        'slope_flow': (flow_segment[-1] - flow_segment[0]) / len(flow_segment),  # Pente du débit
        'integral_flow': np.trapz(flow_segment),  # Intégrale du signal de débit
        'peak_flow_rate': np.max(flow_segment) / len(flow_segment),  # Débit moyen par pic
        'variance_flow': np.var(flow_segment),  # Variance du débit
        'peak_to_mean_ratio_flow': np.max(flow_segment) / np.mean(flow_segment),  # Peak-to-mean ratio
        'cross_corr_flow_pressure': np.max(correlate(flow_segment, pressure_segment)),
        # Corrélation croisée débit-pression
        'derivative_kurtosis_flow': kurtosis(np.diff(flow_segment)),  # Kurtosis de la dérivée du débit
    }

    # Caractéristiques générales du signal
    features['entropy_flow'] = -np.sum(flow_segment * np.log(flow_segment + 1e-6))  # Entropie de Shannon
    features['entropy_pressure'] = -np.sum(pressure_segment * np.log(pressure_segment + 1e-6))  # Entropie de Shannon

    return features


