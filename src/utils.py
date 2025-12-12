import numpy as np
import pandas as pd
import scipy.io
import os
import glob
from scipy.interpolate import interp1d

# ==========================================
# DATA LOADING (from data_loader.py)
# ==========================================

def load_battery_data(mat_file_path):
    """
    Load data from a NASA .mat file.
    
    Args:
        mat_file_path (str): Path to the .mat file.
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a discharge cycle.
              Keys: 'voltage', 'current', 'temp', 'time', 'capacity', 'soh', 'cycle_id', 'battery_id'
              Returns empty list if loading fails or structure is unexpected.
    """
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return []
        
    battery_id = os.path.basename(mat_file_path).split('.')[0]
    
    if battery_id not in mat:
        # print(f"Key {battery_id} not found in {mat_file_path}")
        return []
        
    data = mat[battery_id]
    
    # Structure: data[0][0][0][0] contains 'cycle'
    cycles = data[0][0][0][0]
    
    processed_cycles = []
    
    # Calculate initial capacity (nominal)
    RATED_CAPACITY = 2.0 
    
    cycle_count = 0
    
    # Check shape
    if cycles.ndim == 2:
        num_cycles = cycles.shape[1]
    elif cycles.ndim == 1:
        num_cycles = cycles.shape[0]
    else:
        return []

    for i in range(num_cycles):
        if cycles.ndim == 2:
            cycle = cycles[0][i]
        else:
            cycle = cycles[i]
        type_str = str(cycle['type'][0])
        
        if 'discharge' in type_str:
            data_struct = cycle['data']
            
            # Check for missing data fields
            if data_struct.size == 0:
                continue
                
            # Extract fields
            try:
                voltage = data_struct[0][0]['Voltage_measured'][0]
                current = data_struct[0][0]['Current_measured'][0]
                temp = data_struct[0][0]['Temperature_measured'][0]
                time = data_struct[0][0]['Time'][0]
                capacity = data_struct[0][0]['Capacity'][0][0]
                
                soh = capacity / RATED_CAPACITY
                
                processed_cycles.append({
                    'voltage': voltage,
                    'current': current,
                    'temp': temp,
                    'time': time,
                    'capacity': capacity,
                    'soh': soh,
                    'cycle_id': cycle_count,
                    'battery_id': battery_id
                })
                cycle_count += 1
            except IndexError:
                # Malformed cycle data
                continue
                
    return processed_cycles

def remove_outliers(cycles, threshold=0.05):
    """
    Remove cycles where SOH drops drastically compared to previous cycle,
    indicating sensor error or bad data, but allow for recovery if data stabilizes.
    
    Includes logic to find the longest contiguous segment of valid data.
    """
    if not cycles:
        return []
        
    # Step 1: Filter physically impossible values first
    clean_indices = [i for i, c in enumerate(cycles) if 0 < c['soh'] < 1.1]
    
    if not clean_indices:
        return []
        
    # Step 2: Find longest contiguous sequence of valid cycles 
    segments = []
    current_segment = [cycles[clean_indices[0]]]
    
    for i in range(1, len(clean_indices)):
        curr_idx = clean_indices[i]
        prev_idx = clean_indices[i-1]
        
        curr_cycle = cycles[curr_idx]
        prev_cycle = cycles[prev_idx]
        
        diff = abs(curr_cycle['soh'] - prev_cycle['soh'])
        
        # If consecutive (or close) in time AND SOH change is small
        if (curr_idx - prev_idx <= 5) and (diff < threshold):
            current_segment.append(curr_cycle)
        else:
            # Segment break
            if len(current_segment) > 0:
                segments.append(current_segment)
            current_segment = [curr_cycle]
            
    if len(current_segment) > 0:
        segments.append(current_segment)
        
    # Find longest segment
    if not segments:
        return []
        
    longest_segment = max(segments, key=len)
    
    # Requirement: Segment must be of decent length to be useful
    if len(longest_segment) < 5:
        return []
        
    return longest_segment

def load_all_data(data_dir):
    """
    Load and clean all battery data from directory.
    """
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    all_cycles = []
    
    for f in files:
        cycles = load_battery_data(f)
        cleaned = remove_outliers(cycles)
        all_cycles.extend(cleaned)
        print(f"Loaded {os.path.basename(f)}: {len(cycles)} raw -> {len(cleaned)} cleaned cycles")
        
    return all_cycles


# ==========================================
# PREPROCESSING (from preprocess.py)
# ==========================================

def interpolate_cycle(cycle_data):
    """
    Interpolate cycle data to 1Hz frequency.
    """
    time = cycle_data['time']
    
    if len(time) < 2:
        return None
        
    _, unique_indices = np.unique(time, return_index=True)
    if len(unique_indices) < 2:
        return None
        
    unique_indices.sort()
    
    time = time[unique_indices]
    voltage = cycle_data['voltage'][unique_indices]
    current = cycle_data['current'][unique_indices]
    temp = cycle_data['temp'][unique_indices]
    
    start_time = np.ceil(time[0])
    end_time = np.floor(time[-1])
    
    if start_time >= end_time:
        return None
        
    new_time = np.arange(start_time, end_time + 1, 1.0) # 1Hz
    
    f_v = interp1d(time, voltage, kind='linear', fill_value="extrapolate")
    f_i = interp1d(time, current, kind='linear', fill_value="extrapolate")
    f_t = interp1d(time, temp, kind='linear', fill_value="extrapolate")
    
    new_cycle = {
        'time': new_time,
        'voltage': f_v(new_time),
        'current': f_i(new_time),
        'temp': f_t(new_time),
        'soh': cycle_data['soh'],
        'battery_id': cycle_data.get('battery_id', 'unknown'),
        'cycle_id': cycle_data.get('cycle_id', 0)
    }
    
    return new_cycle

def extract_random_snippet(cycle_data, min_duration=3, max_duration=5):
    """
    Extract a random time snippet from a discharge cycle.
    Assumes cycle_data is already interpolated to 1Hz.
    """
    time = cycle_data['time']
    n_points = len(time)
    
    duration = np.random.randint(min_duration, max_duration + 1)
    
    if n_points < duration:
        return None
        
    start_idx = np.random.randint(0, n_points - duration + 1)
    end_idx = start_idx + duration
    
    snippet = {
        'voltage': cycle_data['voltage'][start_idx:end_idx],
        'current': cycle_data['current'][start_idx:end_idx],
        'temp': cycle_data['temp'][start_idx:end_idx],
        'time_rel': cycle_data['time'][start_idx:end_idx] - cycle_data['time'][start_idx],
        'battery_id': cycle_data['battery_id'],
        'cycle_id': cycle_data['cycle_id']
    }
    
    return snippet

def create_dataset(cycles, snippets_per_cycle=10, min_duration=3, max_duration=5, coverage=None):
    """
    Create a dataset of snippets from a list of cycles (for MLP).
    """
    X = []
    y = []
    
    for cycle in cycles:
        interp_cycle = interpolate_cycle(cycle)
        if interp_cycle is None:
            continue
            
        time_len = len(interp_cycle['time'])
        
        if coverage is not None:
             avg_snippet_len = (min_duration + max_duration) / 2
             if avg_snippet_len == 0: avg_snippet_len = 1
             count = int((time_len * coverage) / avg_snippet_len)
             count = max(1, count)
        else:
             count = snippets_per_cycle
            
        for _ in range(count):
            snippet = extract_random_snippet(interp_cycle, min_duration=min_duration, max_duration=max_duration)
            if snippet is not None:
                X.append(snippet)
                y.append(cycle['soh'])
                
    return X, y

def create_sequence_dataset(cycles, seq_len=10, snippets_per_cycle=10, coverage=None):
    """
    Create a dataset of raw sequences from a list of cycles (for GRU/RNN).
    """
    X_list = []
    y_list = []
    
    for cycle in cycles:
        interp_cycle = interpolate_cycle(cycle)
        if interp_cycle is None:
            continue
            
        time_len = len(interp_cycle['time'])
        
        if coverage is not None:
             count = int((time_len * coverage) / seq_len)
             count = max(1, count)
        else:
             count = snippets_per_cycle
            
        for _ in range(count):
            snippet = extract_random_snippet(interp_cycle, min_duration=seq_len, max_duration=seq_len)
            if snippet is not None:
                # Stack features: Voltage, Current, Temp
                seq = np.stack([
                    snippet['voltage'],
                    snippet['current'],
                    snippet['temp']
                ], axis=1)
                
                X_list.append(seq)
                y_list.append(cycle['soh'])
                
    if not X_list:
        return np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list)


# ==========================================
# FEATURE ENGINEERING (from feature_engineering.py)
# ==========================================

from scipy.stats import linregress

def extract_features(snippet):
    """
    Extract statistical features from a snippet.
    """
    v = snippet['voltage']
    i = snippet['current']
    t = snippet['temp']
    time = snippet['time_rel']
    
    features = {}
    
    # Voltage features
    features['v_mean'] = np.mean(v)
    features['v_std'] = np.std(v)
    features['v_min'] = np.min(v)
    features['v_max'] = np.max(v)
    features['v_range'] = np.max(v) - np.min(v)
    
    # Current features
    features['i_mean'] = np.mean(i)
    features['i_std'] = np.std(i)
    
    # Temperature features
    features['t_mean'] = np.mean(t)
    features['t_std'] = np.std(t)
    features['t_max'] = np.max(t)
    
    # Slopes (Voltage vs Time)
    if len(v) > 1:
        slope, intercept, _, _, _ = linregress(time, v)
        features['v_slope'] = slope
        features['v_intercept'] = intercept
    else:
        features['v_slope'] = 0
        features['v_intercept'] = 0
        
    return features

def prepare_training_data(snippets):
    """
    Convert list of snippets to a DataFrame of features.
    """
    data = []
    for s in snippets:
        data.append(extract_features(s))
        
    return pd.DataFrame(data)
