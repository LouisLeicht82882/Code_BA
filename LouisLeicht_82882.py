# %% [markdown]
# **Responder Groups erstellen**

# %%
# Durchschnittliche Berechnung ESS Score -> Excel

import pandas as pd

df = pd.read_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/labels.csv')

# Funktion zur Bestimmung der Gruppe basierend auf PatientID
def determine_group(patient_id):
    if patient_id.endswith('.1'):
        return 'Pre-Therapy'
    elif patient_id.endswith('.2'):
        return 'During-Therapy'
    elif any(patient_id.endswith(ext) for ext in ['.3', '.4', '.5', '.6', '.7', '.8']):
        return 'Post-Therapy'
    else:
        return 'Unknown'

df['Group'] = df['PatientID'].apply(determine_group)

# Berechnen des durchschnittlichen ESS-Scores für jede Gruppe
average_ess_scores = df.groupby('Group')['ESS_Score'].mean()

print(average_ess_scores)

# %%
# Erstellen Labels mit Excel-Datei

import pandas as pd
import os
import numpy as np

def generate_labels_from_excel_and_file_names(excel_path, output_path, features_dir):
    # Laden der Excel-Datei und Extrahieren der relevanten Spalten
    df_excel = pd.read_excel(excel_path, header=1)
    patient_ess_scores = df_excel[['Unnamed: 0', 'Unnamed: 5']].copy()
    
    patient_ess_scores['Unnamed: 5'] = pd.to_numeric(patient_ess_scores['Unnamed: 5'], errors='coerce')
    
     #Klassifizierung der ESS-Werte
    def classify_ess(score):
        if np.isnan(score):
            return 'missing'
        return 0 if score >= 11 else 1
    
    patient_ess_scores['ESS_Label'] = patient_ess_scores['Unnamed: 5'].apply(classify_ess)
    patient_ess_scores.columns = ['PatientID', 'ESS_Score', 'ESS_Label']

    #Zuordnung der Patientengruppe basierend auf dem Suffix
    def assign_group(patient_id):
        if pd.isnull(patient_id): 
            return 'Unknown'
        parts = patient_id.split('.')
        if len(parts) > 1:
            group_number = parts[1]
            if group_number == '1':
                return 'Pre-Therapy'
            elif group_number == '2':
                return 'During-Therapy'
            else:
                return 'Post-Therapy'
        else:
            return 'Unknown'
    
    patient_ess_scores['Group'] = patient_ess_scores['PatientID'].apply(assign_group)

    # Entfernen von Duplikaten, um sicherzustellen, dass jede PatientID nur einmal vorkommt
    patient_ess_scores.drop_duplicates(subset=['PatientID'], keep='first', inplace=True)
    patient_ess_scores.reset_index(drop=True, inplace=True)
    
    # Entfernen von Zeilen mit fehlenden PatientIDs
    patient_ess_scores.dropna(subset=['PatientID'], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    patient_ess_scores.to_csv(output_path, index=False)
    print(f"Labels and groups saved to {output_path}")

    # Durchlaufen aller Dateien im angegebenen Verzeichnis für Feature-Dateien
    label_files = []
    for file in os.listdir(features_dir):
        if file.endswith('_features.csv'):
            patient_id = os.path.splitext(file)[0].split('_')[0]
            label_row = patient_ess_scores[patient_ess_scores['PatientID'] == patient_id]
            if not label_row.empty:
                ess_label = label_row['ESS_Label'].values[0]
                if ess_label != "missing":
                    label_files.append((file, ess_label))
            else:
                print(f"No label data found for file {file}.")
    
    # Ausgabe der gefundenen Dateien und ihrer zugehörigen Labels
    for file, label in label_files:
        print(f"File: {file} has Label: {label}")

excel_path = '/Volumes/FestPLouis/Excel_Tabelle/Studienkohorte_PSGs.xlsx'
labels_output_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/labels.csv'
features_dir = '/Volumes/FestPLouis/Daten_Schlaflabor_Features'
generate_labels_from_excel_and_file_names(excel_path, labels_output_path, features_dir)

# %%
#Therapie-Stand hinzufügen anhand von Präfix (Pre, During,Post)

import pandas as pd
import os

labels_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/labels.csv'
responder_labels_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_labels.csv'
output_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder.csv'

labels_df = pd.read_csv(labels_path)
responder_labels_df = pd.read_csv(responder_labels_path)

labels_df['Patienten ID'] = labels_df['PatientID'].apply(lambda x: str(x).split('.')[0])

# Nur nicht-NaN Werte für PatientID_Pre verwenden
responder_labels_df = responder_labels_df.dropna(subset=['PatientID_Pre'])
responder_labels_df['Patienten ID'] = responder_labels_df['PatientID_Pre'].apply(lambda x: str(x).split('.')[0])

# Responder-Tag zuweisen und in numerische Werte umwandeln
responder_labels_df['Responder'] = responder_labels_df['Responder'].map({'good': 1, 'bad': 0})
responder_map = responder_labels_df.set_index('Patienten ID')['Responder'].to_dict()

labels_df['Responder'] = labels_df['Patienten ID'].map(responder_map)

print(labels_df)

labels_df.to_csv(output_path, index=False)

# %%
# Erstellen der Labels (good/bad)
import pandas as pd
import numpy as np

labels_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/labels.csv'
df = pd.read_csv(labels_path)

df['ESS_Score'] = pd.to_numeric(df['ESS_Score'], errors='coerce')

# Präfix und Suffix der Patienten-ID extrahieren
def extract_prefix_suffix(patient_id):
    try:
        prefix, suffix = patient_id.split('.')
        return prefix, int(suffix)
    except ValueError:
        return np.nan, np.nan

df[['Patient', 'PatientID_Suffix']] = df['PatientID'].apply(lambda x: pd.Series(extract_prefix_suffix(x)))

df = df.dropna(subset=['Patient', 'PatientID_Suffix'])

# Filter für Pre-Therapy und Post-Therapy
pre_therapy_df = df[df['Group'] == 'Pre-Therapy']
post_therapy_df = df[df['Group'] == 'Post-Therapy']

# Letzte Post-Therapy-Werte finden, die nicht 'missing' sind
post_therapy_df = post_therapy_df[post_therapy_df['ESS_Label'] != 'missing']
post_therapy_last_df = post_therapy_df.loc[post_therapy_df.groupby('Patient')['PatientID_Suffix'].idxmax()]

# Merge von Pre-Therapy und letzten Post-Therapy Daten
merged_df = pd.merge(pre_therapy_df[['Patient', 'ESS_Score', 'PatientID']], 
                     post_therapy_last_df[['Patient', 'ESS_Score', 'PatientID']], 
                     on='Patient', 
                     suffixes=('_Pre', '_Post'))

#Bestimmen ob Responder good/bad
merged_df['Responder'] = np.where((merged_df['ESS_Score_Post'] <= 10) & 
                                  ((merged_df['ESS_Score_Pre'] - merged_df['ESS_Score_Post']) >= 2), 
                                  'good', 'bad')

print(merged_df)

updated_labels_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_labels.csv'
merged_df.to_csv(updated_labels_path, index=False)

# %%
#Responder Datei erstellen

import pandas as pd

csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder.csv'

df = pd.read_csv(csv_path)

# Filterung der Daten: Nur die PatientIDs mit dem Suffix '.1' und einem gültigen Responder-Tag (0.0 oder 1.0)
filtered_df = df[(df['PatientID'].str.endswith('.1')) & (df['Responder'].isin([0.0, 1.0]))]

output_csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv'
filtered_df.to_csv(output_csv_path, index=False)

print(f"Gefilterte Daten wurden in {output_csv_path} gespeichert.")

# %%
#Erstellen CSV-für Pre-Therapie Group Responders

import os
import pandas as pd
import xml.etree.ElementTree as ET

csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv'
data_dir = '/Volumes/FestPLouis/Daten_Schlaflabor'
output_csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/relevant_folders.csv'

df = pd.read_csv(csv_path)

df['PatientID'] = df['PatientID'].astype(str)
df = df[df['PatientID'].notna()]

patient_ids = df['PatientID'].unique()

print("Liste der PatientIDs:")
for patient_id in patient_ids:
    print(patient_id)

# Methodezum extrahiert der PatientID aus der RML-Datei
def extract_patient_id_from_rml(rml_file_path):
    try:
        tree = ET.parse(rml_file_path)
        root = tree.getroot()
        ns = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
        patient_element = root.find('.//ns:PatientID', ns)
        if patient_element is not None:
            return patient_element.text
    except ET.ParseError as e:
        print(f"Fehler beim Parsen der RML-Datei {rml_file_path}: {e}")
    return None

relevant_data = []

# Durchsuche das Verzeichnis nach Ordnern und überprüfe die RML-Dateien
for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        # Prüfen, ob es eine RML-Datei im Ordner gibt
        rml_files = [file for file in os.listdir(dir_path) if file.endswith('.rml')]
        if not rml_files:
            print(f"Keine RML-Datei im Ordner {dir_path} gefunden.")
            continue
        for rml_file in rml_files:
            rml_file_path = os.path.join(dir_path, rml_file)
            rml_patient_id = extract_patient_id_from_rml(rml_file_path)
            if rml_patient_id:
                print(f"Gefundene PatientID in RML-Datei: {rml_patient_id}")
                if rml_patient_id in patient_ids:
                    relevant_data.append((rml_patient_id, dir_path))
                    print(f"Relevanter Ordner gefunden: {dir_path}")
                    break  
            else:
                print(f"PatientID konnte in der RML-Datei {rml_file_path} nicht gefunden werden.")

if relevant_data:
    df_relevant = pd.DataFrame(relevant_data, columns=['PatientID', 'Folder'])
    df_relevant.sort_values(by='PatientID', inplace=True)  # Sortieren nach PatientID
    df_relevant.to_csv(output_csv_path, index=False)
    print(f"Relevante Daten wurden in {output_csv_path} gespeichert.")
else:
    print("Es wurden keine relevanten Ordner gefunden.")

print("Prozess abgeschlossen.")

# %%
#sortieren nach Patienten ID

import pandas as pd

input_csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/relevant_folders.csv'

output_csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/relevant_folders.csv'

df = pd.read_csv(input_csv_path)

df['PatientID'] = df['PatientID'].astype(str)

# Aufspalten der PatientID zum sortieren
def split_patient_id(patient_id):
    parts = patient_id.split('.')
    return int(parts[0]), int(parts[1])

# Sortieren basierend auf den aufgesplitteten Werten
df_sorted = df.sort_values(by='PatientID', key=lambda x: x.apply(split_patient_id))

df_sorted.to_csv(output_csv_path, index=False)

print(f"Die CSV-Datei wurde nach PatientID im Format '1.1', '2.1', ... sortiert und unter {output_csv_path} gespeichert.")

# %% [markdown]
# **Data Pre-processing**

# %%
#Extrahieren Samplin Frequenz

import pyedflib

def get_sampling_frequency(edf_file_path):
    try:
        edf = pyedflib.EdfReader(edf_file_path)
        fs = edf.getSampleFrequency(0)
        edf.close()
        return fs
    except Exception as e:
        print(f"Fehler beim Lesen der EDF-Datei: {e}")
        return None

edf_file_path = '/Volumes/FestPLouis/Daten_Schlaflabor/00000029-A5BS00755/00000029-A5BS00755[003].edf'

sampling_frequency = get_sampling_frequency(edf_file_path)
if sampling_frequency:
    print(f"Die Sampling-Frequenz beträgt: {sampling_frequency} Hz")
else:
    print("Die Sampling-Frequenz konnte nicht extrahiert werden.")

# %%
#Bereinigung der Pre-Therapy Daten -> relevant_folders.csv

import os
import numpy as np
import pandas as pd
import pyedflib
import xml.etree.ElementTree as ET
from scipy.signal import firwin, lfilter
from sklearn.decomposition import FastICA

def load_eeg_from_psg(file_path, channel_labels, fs=100.00):
    # Lädt EEG-Daten aus einer PSG (EDF)-Datei basierend auf den angegebenen Kanallabels
    eeg_data = []
    edf = None
    try:
        edf = pyedflib.EdfReader(file_path) # Öffnet die EDF-Datei
        available_channels = edf.getSignalLabels() # Liest die verfügbaren Kanallabels
        
        channel_indices = [available_channels.index(label) for label in channel_labels if label in available_channels]
        
        if not channel_indices:
            raise ValueError("Keine der angegebenen Kanalbezeichnungen wurden in der Datei gefunden.")
        
        for index in channel_indices:
            signal = edf.readSignal(index) # Liest Signale der ausgewählten Kanäle
            eeg_data.append(signal)
        
        if not eeg_data:
            raise ValueError("Keine EEG-Daten extrahiert.")
        
        eeg_data = np.array(eeg_data)
    finally:
        if edf is not None:
            edf.close()

    return eeg_data

def apply_fir_filter(eeg_data, fs=100.00, lowcut=0.5, highcut=40.0):
     #Anwenden eines FIR-Bandpassfilter 
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if low >= high or low <= 0 or high >= 1:
        raise ValueError(f"Invalid cutoff frequencies: {lowcut, highcut} for fs={fs}.")
    
    taps = firwin(numtaps=101, cutoff=[low, high], pass_zero=False)
    filtered_data = lfilter(taps, 1.0, eeg_data, axis=1)
    return filtered_data

def preprocess_eeg_data(eeg_data, fs=100.00):
    eeg_filtered = apply_fir_filter(eeg_data, fs)
    
    #Bereinigung der Daten durch ICA (Entfernen Artefakte)
    ica = FastICA(n_components=eeg_filtered.shape[0])
    eeg_ica = ica.fit_transform(eeg_filtered.T).T

    return eeg_ica

def extract_sleep_stages(rml_file_path):
    # Extrahieren der Schlafstadien aus einer RML-Datei
    sleep_stages = []
    tree = ET.parse(rml_file_path)
    root = tree.getroot()

    ns = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}

    for staging_type in ['UserStaging', 'MachineStaging']:
        staging_root = root.find(f".//ns:{staging_type}/ns:NeuroRKStaging", ns)
        if staging_root is not None:
            for stage in staging_root.findall("ns:Stage", ns):
                stage_type = stage.attrib.get('Type', 'NotScored')
                start_time = int(stage.attrib.get('Start', 0))
                sleep_stages.append((stage_type, start_time))

    sleep_stages.sort(key=lambda x: x[1])

    return sleep_stages

def assign_stages_to_segments(sleep_stages, num_segments, segment_duration):
    #Zuweisen der Schlafstadium (Zu jedem Segment)
    segment_stages = []
    current_stage = None
    stage_index = 0

    for i in range(num_segments):
        segment_start_time = i * segment_duration
        if stage_index < len(sleep_stages) - 1 and segment_start_time >= sleep_stages[stage_index + 1][1]:
            stage_index += 1
        current_stage = sleep_stages[stage_index][0]
        segment_stages.append(current_stage)

    return segment_stages

def extract_patient_id_from_csv(folder_path, relevant_folders_df):
    patient_id_row = relevant_folders_df[relevant_folders_df['Folder'] == folder_path]
    
    if not patient_id_row.empty:
        patient_id = patient_id_row['PatientID'].values[0]
        return patient_id
    else:
        return None
    
# Funktion zum verarbeiten der einzelnen PSG-Dateien
def process_single_file(folder_path, sleep_stages, output_dir, channel_labels, relevant_folders_df, fs=100.00, segment_duration=30):
    try:
        patient_id = extract_patient_id_from_csv(folder_path, relevant_folders_df)
        
        if not patient_id:
            raise ValueError(f"Keine Patienten-ID für den Ordner {folder_path} gefunden.")
        
        # Suche nach der EDF-Datei im Ordner
        edf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".edf")]
        if not edf_files:
            raise ValueError(f"Keine EDF-Datei im Ordner {folder_path} gefunden.")
        
        # Lade die EEG-Daten aus der EDF-Datei
        eeg_data = load_eeg_from_psg(edf_files[0], channel_labels, fs)
        
        if eeg_data.size == 0:
            raise ValueError(f"Keine EEG-Daten im Ordner {folder_path} extrahiert.")
        
        eeg_preprocessed = preprocess_eeg_data(eeg_data, fs)
        
        # Bestimme die Anzahl der Segmente
        num_segments = eeg_preprocessed.shape[1] // (segment_duration * int(fs))
        
        # Weisen Sie den Segmenten die Schlafstadien zu
        segment_stages = assign_stages_to_segments(sleep_stages, num_segments, segment_duration)
        
        # Mapping der Schlafstadien zu ihren Bezeichnungen
        stage_names = {
            0: "wake",
            1: "stage1",
            2: "stage2",
            3: "stage3",
            4: "stage4",
            5: "rem"
        }
        
        # Speichere die Segmente und die zugehörigen Schlafstadien
        for i in range(num_segments):
            segment = eeg_preprocessed[:, i * segment_duration * int(fs): (i + 1) * segment_duration * int(fs)]
            stage = segment_stages[i]
            stage_name = stage_names.get(stage, f"stage{stage}")  # Verwende die exakte Bezeichnung des Schlafstadiums
            
            segment_file = os.path.join(output_dir, f"{patient_id}_segment{i+1}_{stage_name}.npy")
            
            np.save(segment_file, segment)
        
        return True
    except Exception as e:
        print(f"Fehler beim Verarbeiten des Ordners {folder_path}: {e}")
        return False


def process_relevant_folders(relevant_folders_df, base_dir, output_dir, channel_labels, fs=100.00, segment_duration=30):
    # Verarbeitet alle relevanten Ordner, um EEG-Segmente zu extrahieren (30Sek) und zu speichern
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    
    for folder_path in relevant_folders_df['Folder']:
        if not os.path.exists(folder_path):
            continue

        rml_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".rml")]
        if not rml_files:
            continue
        
        sleep_stages = extract_sleep_stages(rml_files[0])

        result = process_single_file(folder_path, sleep_stages, output_dir, channel_labels, relevant_folders_df, fs, segment_duration)
        results.append(result)
    
    successful = len([result for result in results if result])
    return successful, len(results) - successful

base_dir = '/Volumes/FestPLouis/Daten_Schlaflabor'
output_dir = '/Volumes/FestPLouis/Daten_Schlaflabor_Segmente'
csv_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/relevant_folders.csv'
channel_labels = ['EEG C4-A1', 'EEG C3-A2', 'EEG A1-A2']  
segment_duration = 30  # Segmentdauer in Sekunden

relevant_folders_df = pd.read_csv(csv_path)

successful, failed = process_relevant_folders(relevant_folders_df, base_dir, output_dir, channel_labels, fs=100.00, segment_duration=segment_duration)

print(f"Erfolgreich verarbeitete Ordner: {successful}")
print(f"Fehlgeschlagene Ordner: {failed}")

# %% [markdown]
# **Feature Engineering**

# %%
# Feature Extraktion

import os
import numpy as np
import pywt
import pandas as pd
from scipy.signal import butter, lfilter, welch, find_peaks
from scipy.stats import entropy, kurtosis, skew
from joblib import Parallel, delayed, cpu_count
import antropy as ant

# Funktion zur Erkennung von Schlafspindeln (12-16 Hz) im N2-Stadium
def detect_sleep_spindles(eeg_data, fs, threshold_factor=2):
    # Berechne die Standardabweichung des Signals
    std = np.std(eeg_data)  
    # Finde Peaks im gefilterten Signal, die über einem bestimmten Schwellenwert liegen
    peaks, _ = find_peaks(eeg_data, height=std * threshold_factor)
    # Die Anzahl der gefundenen Peaks entspricht der Anzahl der Schlafspindeln
    return len(peaks)

# Funktion zur Durchführung der Wavelet-Transformation
def wavelet_transform(eeg_data, wavelet='db4', level=5):
    coeffs = pywt.wavedec(eeg_data, wavelet, level=level)
    return coeffs

# Funktion zur Erkennung von K-Komplexen im N2-Stadium 
def detect_k_complexes(eeg_data, fs):
    peaks, _ = find_peaks(eeg_data, height=np.std(eeg_data) * 3, distance=int(0.5 * fs)) # Schwellenwert für K-Komplexe
    return len(peaks)  

# Funktion zur Berechnung der Shannon-Entropie
def shannon_entropy(data):
    return entropy(np.histogram(data, bins=100)[0], base=2)

# Funktion zur Berechnung der Tsallis-Entropie
def tsallis_entropy(data, q=2):
    data = np.histogram(data, bins=100)[0]
    prob = data / np.sum(data)
    return (1 - np.sum(prob ** q)) / (q - 1)

# Funktion zur Berechnung der Renyi-Entropie
def renyi_entropy(data, alpha=2):
    data = np.histogram(data, bins=100)[0]
    prob = data / np.sum(data)
    return 1 / (1 - alpha) * np.log(np.sum(prob ** alpha))

# Funktion zur Berechnung der Log Energy Entropy
def log_energy_entropy(data):
    return np.sum(np.log(data**2))

# Definition der Hjorth-Parameter und Teager-Energie-Funktion
def hjorth_activity(data):
    return np.var(data)

def hjorth_mobility(data):
    return np.sqrt(np.var(np.diff(data)) / np.var(data))

def hjorth_complexity(data):
    return hjorth_mobility(np.diff(data)) / np.var(data)

def mean_teager_energy(data):
    return np.mean(data[:-2]**2 - data[1:-1] * data[2:])

# Funktion zur Berechnung der Amplituden und Latenzen für eine EEG-Epoche
def calculate_amplitude_latency(eeg_epoch, sf):
    results = {}

    # Berechne die Peak-to-Peak Amplitude
    peak_to_peak_amplitude = np.max(eeg_epoch) - np.min(eeg_epoch)
    results['peak_to_peak_amplitude'] = peak_to_peak_amplitude
    
    # Finde Peaks und deren Latenzen
    peaks, _ = find_peaks(eeg_epoch)
    troughs, _ = find_peaks(-eeg_epoch)
    
    if peaks.size > 0:
        peak_amplitude = np.max(eeg_epoch[peaks])
        peak_latency = peaks[np.argmax(eeg_epoch[peaks])] / sf  
        results['peak_amplitude'] = peak_amplitude
        results['peak_latency'] = peak_latency

    if troughs.size > 0:
        trough_amplitude = np.min(eeg_epoch[troughs])
        trough_latency = troughs[np.argmin(eeg_epoch[troughs])] / sf  
        results['trough_amplitude'] = trough_amplitude
        results['trough_latency'] = trough_latency

    return results

# Spektralanalyse und Feature-Extraktion mit der Welch-Methode
def perform_welch_and_extract_features(eeg_data, fs=173.61, nperseg=256, channel_label="", sleep_stage=""):
    features = []
    feature_names = []
    
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "sigma": (12, 16),  
        "beta": (16, 30),
        "gamma": (30, 40),
    }
    
    for ch in eeg_data:
        freqs, psd = welch(ch, fs=fs, nperseg=nperseg)
        
        # Berechne und speichere die Power und andere Features für jedes Band
        for band, (low_freq, high_freq) in bands.items():
            band_psd = psd[(freqs >= low_freq) & (freqs <= high_freq)]
            band_power = np.sum(band_psd)
            features.append(band_power)
            feature_names.append(f"{channel_label}_{band}_power")

            # Entropien und Komplexitätsmaße für das Frequenzband
            features.append(shannon_entropy(band_psd))
            feature_names.append(f"{channel_label}_{band}_shannon_entropy")
            
            features.append(tsallis_entropy(band_psd))
            feature_names.append(f"{channel_label}_{band}_tsallis_entropy")
            
            features.append(renyi_entropy(band_psd))
            feature_names.append(f"{channel_label}_{band}_renyi_entropy")
            
            features.append(log_energy_entropy(band_psd))
            feature_names.append(f"{channel_label}_{band}_log_energy_entropy")
            
            features.append(ant.svd_entropy(band_psd))
            feature_names.append(f"{channel_label}_{band}_svd_entropy")

        # Analyse für Delta-Wellen im N3-Stadium
        if sleep_stage == "Stage3":
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
            features.append(delta_power)
            feature_names.append(f"{channel_label}_delta_power")

        # Verhältnis zwischen Alpha- und Delta-Power zur Unterscheidung von Wach- und Schlafzuständen
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        if alpha_power > delta_power:
            state = 1  # Wachzustand
        else:
            state = 0  # Schlafzustand
        
        features.append(state)
        feature_names.append(f"{channel_label}_sleep_wake_state")

    return features, feature_names

# Funktion zur Extraktion der Merkmale
def extract_features(data, sf, filename="", channel_labels=None):
    features = []
    feature_names = []
    
    if "Stage1" in filename:
        sleep_stage = "Stage1"
    elif "Stage2" in filename:
        sleep_stage = "Stage2"
    elif "Stage3" in filename:
        sleep_stage = "Stage3"
    elif "Wake" in filename:
        sleep_stage = "Wake"
    elif "REM" in filename:
        sleep_stage = "REM"
    else:
        sleep_stage = "other"
    
    for i, ch in enumerate(data):
        ch_label = channel_labels[i] if channel_labels else f"ch_{i}"
        ch_features = []
        ch_feature_names = []

        # Berechnung von Amplitude und Latenz für die Epoche
        amp_lat_features = calculate_amplitude_latency(ch, sf)
        for feature_name, feature_value in amp_lat_features.items():
            ch_features.append(feature_value)
            ch_feature_names.append(f"{ch_label}_{feature_name}")

        # Statistische Merkmale
        ch_features.append(np.mean(ch))        # Mittelwert
        ch_feature_names.append(f"{ch_label}_mean")
        ch_features.append(np.median(ch))      # Median
        ch_feature_names.append(f"{ch_label}_median")
        ch_features.append(np.var(ch))    # Varianz
        ch_feature_names.append(f"{ch_label}_variance")
        ch_features.append(np.std(ch))         # Standardabweichung
        ch_feature_names.append(f"{ch_label}_std_deviation")
        ch_features.append(np.max(ch))         # Maximum
        ch_feature_names.append(f"{ch_label}_max_value")
        ch_features.append(np.min(ch))         # Minimum
        ch_feature_names.append(f"{ch_label}_min_value")
        ch_features.append(np.percentile(ch, 25))  # 25. Perzentil
        ch_feature_names.append(f"{ch_label}_percentile_25")
        ch_features.append(np.percentile(ch, 75))  # 75. Perzentil
        ch_feature_names.append(f"{ch_label}_percentile_75")

        # Wavelet-Transformation
        coeffs = wavelet_transform(ch)
        for j, coeff in enumerate(coeffs):
            ch_features.append(np.mean(coeff))
            ch_feature_names.append(f"{ch_label}_wavelet_mean_{j}")
            ch_features.append(np.var(coeff))
            ch_feature_names.append(f"{ch_label}_wavelet_var_{j}")

        ch_features.append(hjorth_activity(ch))
        ch_feature_names.append(f"{ch_label}_hjorth_activity")
        ch_features.append(hjorth_mobility(ch))
        ch_feature_names.append(f"{ch_label}_hjorth_mobility")
        ch_features.append(hjorth_complexity(ch))
        ch_feature_names.append(f"{ch_label}_hjorth_complexity")
        ch_features.append(mean_teager_energy(ch))
        ch_feature_names.append(f"{ch_label}_mean_teager_energy")
        
        ch_features.append(shannon_entropy(ch))
        ch_feature_names.append(f"{ch_label}_shannon_entropy")
        ch_features.append(tsallis_entropy(ch))
        ch_feature_names.append(f"{ch_label}_tsallis_entropy")
        ch_features.append(renyi_entropy(ch))
        ch_feature_names.append(f"{ch_label}_renyi_entropy")
        ch_features.append(log_energy_entropy(ch))
        ch_feature_names.append(f"{ch_label}_log_energy_entropy")

        ch_features.append(kurtosis(ch))
        ch_feature_names.append(f"{ch_label}_kurtosis")
        ch_features.append(skew(ch))
        ch_feature_names.append(f"{ch_label}_skewness")
        ch_features.append(ant.svd_entropy(ch))
        ch_feature_names.append(f"{ch_label}_svd_entropy")

        # Füge Welch-Features hinzu und berücksichtige das Schlafstadium
        ch_welch_features, ch_welch_feature_names = perform_welch_and_extract_features([ch], sf, channel_label=ch_label, sleep_stage=sleep_stage)
        ch_features.extend(ch_welch_features)
        ch_feature_names.extend(ch_welch_feature_names)
        
        # Spezifische Erkennung von Schlafspindeln und K-Komplexen im N2-Stadium
        if sleep_stage == "Stage2":
            spindle_count = detect_sleep_spindles(ch, sf)
            k_complex_count = detect_k_complexes(ch, sf)
            ch_features.append(spindle_count)
            ch_feature_names.append(f"{ch_label}_spindle_count")
            ch_features.append(k_complex_count)
            ch_feature_names.append(f"{ch_label}_k_complex_count")

        features.extend(ch_features)
        feature_names.extend(ch_feature_names)
    
    return features, len(features) // len(data), feature_names

import pandas as pd

def process_file(file_path, output_dir, channel_labels):
    file_name = os.path.basename(file_path)
    
    
    patient_id = file_name.split('_')[0] 
    segment_number = file_name.split('_')[1]  
    
    # Erstellen eines neuen Dateinamens
    stage = file_name.split('_')[-1].replace('stage', '').replace('.npy', '')  # 'Wake'
    new_file_name = f"{patient_id}_{segment_number}_{stage}.csv"
    
    session_features = []
    print(f"    Datei gefunden: {file_path}")
    data = np.load(file_path)
    sf = 128  
    print(f"    Datenform: {data.shape}")
    print(f"    Datentyp: {data.dtype}")
    
    if data.ndim == 2:
        try:
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            features, num_features, feature_names = extract_features(data, sf, filename=file_path, channel_labels=channel_labels)
            
            # Füge das Schlafstadium als zusätzliches Feature hinzu
            features.append(stage)
            feature_names.append("SleepStage")
            
            # Speicherung als DataFrame
            df = pd.DataFrame([features], columns=feature_names)
            output_file = os.path.join(output_dir, new_file_name)
            df.to_csv(output_file, index=False)
            print(f"    Merkmale gespeichert: {output_file}")
            return True, None
        except Exception as e:
            print(f"    Fehler bei Datei {file_path}: {str(e)}")
            return False, file_path
    else:
        print(f"    Fehler bei Datei {file_path}: Datenformat nicht kompatibel")
        return False, file_path
        
# Funktion zur Verarbeitung aller Dateien in einem Verzeichnis
def process_all_files(data_dir, output_dir, channel_labels):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = [os.path.join(root, file_name) for root, dirs, files in os.walk(data_dir) for file_name in files if file_name.endswith('.npy')]

    print(f"Anzahl der gefundenen Dateien: {len(file_paths)}")

    print(f"Gefundene Dateien: {file_paths}")  

    results = Parallel(n_jobs=cpu_count(), backend="loky")(delayed(process_file)(file_path, output_dir, channel_labels) for file_path in file_paths)

    total_success = sum(result[0] for result in results)
    failed_files = [result[1] for result in results if result[1] is not None]

    print("\nVerarbeitung abgeschlossen.")
    print(f"Erfolgreich verarbeitete Dateien: {total_success}")
    print(f"Dateien mit Fehlern: {failed_files}")

data_dir = "/Volumes/FestPLouis/Daten_Schlaflabor_Segmente"
output_dir = "/Volumes/FestPLouis/Daten_Schlaflabor_Segmente_Features"
channel_labels = ['EEG C4-A1', 'EEG C3-A2','EEG A1-A2'] 

process_all_files(data_dir, output_dir, channel_labels)

# %% [markdown]
# **Kombinieren und Überpüfen Featuress**

# %%
#Kombinieren der Features zu einer Datei

import os
import pandas as pd
import numpy as np

features_directory = '/Volumes/FestPLouis/Daten_Schlaflabor_Segmente_Features'

all_files = [os.path.join(features_directory, f) for f in os.listdir(features_directory) if f.endswith('.csv')]

dfs = []

# Schlafstadien-Kodierung
sleep_stage_mapping = {
    'Wake': 0,
    'Stage1': 1,
    'Stage2': 2,
    'Stage3': 3,
    'Stage4': 4,
    'REM': 5
}

# Alle Dateien einlesen und die PatientID sowie kodierte Schlafstadien hinzufügen
for file in all_files:
    df = pd.read_csv(file)
    
    # PatientID aus dem Dateinamen extrahieren 
    patient_id = os.path.basename(file).split('_')[0]
    df['PatientID'] = patient_id  
    
    # Schlafstadien kodieren
    if 'SleepStage' in df.columns:
        df['SleepStage'] = df['SleepStage'].map(sleep_stage_mapping).fillna(df['SleepStage'])
    
    # Für Stage2 und Stage3 spezielle Features hinzufügen
    if 'SleepStage' in df.columns:
        # Für Stage2: Zusätzliche Spalten
        stage2_features = ['EEG C4-A1_spindle_count', 'EEG C4-A1_k_complex_count', 'EEG C3-A2_spindle_count', 'EEG C3-A2_k_complex_count', 'EEG A1-A2_spindle_count', 'EEG A1-A2_k_complex_count']
        if any(feature not in df.columns for feature in stage2_features):
            df[stage2_features] = np.nan  # fehlende Spalten auf NaN
        
        # Für Stage3: Delta-Power-Features
        delta_features = ['EEG C4-A1_delta_power', 'EEG C3-A2_delta_power', 'EEG A1-A2_delta_power']
        if any(feature not in df.columns for feature in delta_features):
            df[delta_features] = np.nan  # fehlende Spalten auf NaN

    dfs.append(df)


features_df = pd.concat(dfs, ignore_index=True)


labels_df = pd.read_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv')

# Konvertiere die PatientID in beiden DataFrames in den gleichen Datentyp (z.B. string)
features_df['PatientID'] = features_df['PatientID'].astype(str)
labels_df['PatientID'] = labels_df['PatientID'].astype(str)

# Zusammenführen von features_df und labels_df basierend auf der PatientID-Spalte
combined_df = pd.merge(features_df, labels_df, how='left', on='PatientID')

# Füllen von NaN-Werten nur in numerischen Spalten
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())

numeric_df = combined_df.select_dtypes(include=[np.number])

mandatory_features = ['EEG C4-A1_spindle_count', 'EEG C4-A1_k_complex_count', 'EEG C3-A2_spindle_count', 'EEG C3-A2_k_complex_count', 'EEG A1-A2_spindle_count', 'EEG A1-A2_k_complex_count', 'EEG C4-A1_delta_power', 'EEG C3-A2_delta_power', 'EEG A1-A2_delta_power']
for feature in mandatory_features:
    if feature not in numeric_df.columns:
        print(f"Fehlendes Feature: {feature}")

numeric_df.to_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/kombinierte_Merkmale.csv', index=False)

print("Alle Features wurden erfolgreich kombiniert und gespeichert.")

# %%
# WIe vile Spalten/Features pro Datei in der kombinierten Datei (ohne Responder, Patienten ID, ESS_Score, SleepStage)

import pandas as pd

data = pd.read_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/kombinierte_Merkmale.csv')

feature_columns = data.drop(columns=['Responder', 'Patienten ID', 'ESS_Score', 'SleepStage'], errors='ignore')

print(f"Total number of features: {feature_columns.shape[1]}")

# %% [markdown]
# **Maschinelles Lernen**

# %% [markdown]
# **XGBoost**

# %%
#XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap
import seaborn as sns

features_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/kombinierte_Merkmale.csv'
responder_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv'

data = pd.read_csv(features_path)

# Überprüfen, ob 'Patienten ID' als Ganzzahl oder Zeichenfolge formatiert ist und konvertieren
if data['Patienten ID'].dtype != 'object':
    data['Patienten ID'] = data['Patienten ID'].astype(str)

responder_data = pd.read_csv(responder_path)

# Konvertiere 'Patienten ID' in beiden DataFrames in den gleichen Datentyp
responder_data['Patienten ID'] = responder_data['Patienten ID'].astype(str)

# Zusammenführen der Features und Responder-Labels basierend auf der 'Patienten ID'
combined_df = pd.merge(data, responder_data[['Patienten ID', 'Responder']], on='Patienten ID')

if 'Responder_x' in combined_df.columns and 'Responder_y' in combined_df.columns:
    combined_df['Responder'] = combined_df['Responder_x']  # oder 'Responder_y', je nachdem, welche Sie behalten möchten
    combined_df = combined_df.drop(columns=['Responder_x', 'Responder_y'])

# Überprüfen, ob die Spalte 'Responder' nach der Bereinigung vorhanden ist
if 'Responder' not in combined_df.columns:
    raise KeyError("Die 'Responder'-Spalte fehlt nach der Zusammenführung und Bereinigung in 'combined_df'.")

# Entfernen der 'Patienten ID', 'ESS_Score' und 'Responder' aus den Features
X = combined_df.drop(columns=['Patienten ID', 'ESS_Score', 'Responder'], errors='ignore')
y = combined_df['Responder']

# Sicherstellen, dass keine NaNs oder unendlichen Werte in den Daten vorhanden sind
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Berechnung des scale_pos_weight basierend auf den Klassenverteilungen
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Modelltraining unter Berücksichtigung des Klassenungleichgewichts
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.05],
    'reg_alpha': [2, 3],
    'reg_lambda': [4, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# GridSearchCV zur Modellsuche
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)
y_proba = best_xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
classification_rep = classification_report(y_test, y_pred)

print("Ergebnisse für das optimierte XGBoost-Modell:")
print(f"Beste Parameter: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print(classification_rep)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"Spezifität (Specificity): {specificity}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity}")

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad Responder', 'Good Responder'], yticklabels=['Bad Responder', 'Good Responder'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix XGBoost')
plt.show()

importances = best_xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

print("Top Features nach Wichtigkeit sortiert:")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]}")

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# SHAP-Explainer erstellen
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_test)

# Global Interpretability Plot - Übersicht über die Feature Importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Global Interpretability Plot - Detaillierte Übersicht über die Auswirkungen
shap.summary_plot(shap_values, X_test)

# Local Interpretability Plot - Analyse einer spezifischen Vorhersage
sample_idx = 0  
shap.force_plot(explainer.expected_value, shap_values[sample_idx, :], X_test.iloc[sample_idx, :], matplotlib=True)

# %%
#Vergleich Training Test

#Berechnung und Anzeige der Metriken auf den Testdaten
print("Ergebnisse für das optimierte XGBoost-Modell auf den Testdaten:")
y_pred_test = best_xgb_model.predict(X_test)
y_proba_test = best_xgb_model.predict_proba(X_test)[:, 1]

accuracy_test = accuracy_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_proba_test)
classification_rep_test = classification_report(y_test, y_pred_test)
cm_test = confusion_matrix(y_test, y_pred_test)

tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
specificity_test = tn_test / (tn_test + fp_test)
sensitivity_test = tp_test / (tp_test + fn_test)

print(f"Accuracy: {accuracy_test}")
print(f"ROC-AUC: {roc_auc_test}")
print(f"Spezifität (Specificity): {specificity_test}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity_test}")
print(classification_rep_test)

# Berechnung und Anzeige der Metriken auf den Trainingsdaten
print("\nErgebnisse für das optimierte XGBoost-Modell auf den Trainingsdaten:")
y_pred_train = best_xgb_model.predict(X_train)
y_proba_train = best_xgb_model.predict_proba(X_train)[:, 1]

accuracy_train = accuracy_score(y_train, y_pred_train)
roc_auc_train = roc_auc_score(y_train, y_proba_train)
classification_rep_train = classification_report(y_train, y_pred_train)
cm_train = confusion_matrix(y_train, y_pred_train)

tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
specificity_train = tn_train / (tn_train + fp_train)
sensitivity_train = tp_train / (tp_train + fn_train)

print(f"Accuracy: {accuracy_train}")
print(f"ROC-AUC: {roc_auc_train}")
print(f"Spezifität (Specificity): {specificity_train}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity_train}")
print(classification_rep_train)

# Vergleich der Metriken auf Trainings- und Testdaten
print("\nVergleich der Ergebnisse (Trainingsdaten vs. Testdaten):")
print(f"Accuracy - Training: {accuracy_train}, Test: {accuracy_test}")
print(f"ROC-AUC - Training: {roc_auc_train}, Test: {roc_auc_test}")
print(f"Spezifität - Training: {specificity_train}, Test: {specificity_test}")
print(f"Sensitivität - Training: {sensitivity_train}, Test: {sensitivity_test}")

# %%
# Berechnung der SHAP-Interaktionswerte
shap_interaction_values = explainer.shap_interaction_values(X_test)

# Mittelwerte der Interaktionswerte berechnen
interaction_importance = shap_interaction_values.mean(axis=0)

interaction_importance_df = pd.DataFrame(interaction_importance, index=X_test.columns, columns=X_test.columns)

np.fill_diagonal(interaction_importance_df.values, 0)

# Entfache alle Paarungen in eine Spalte, um alle Interaktionen zu sehen
all_interactions = interaction_importance_df.unstack()

sorted_interactions = all_interactions.sort_values(ascending=False)

sorted_interactions = sorted_interactions[~sorted_interactions.index.duplicated(keep='first')]

print("Alle Feature-Interaktionen, sortiert nach Stärke:")
print(sorted_interactions)

# %%
# Abhängigkeitspolot für die stärkste Wechselwirkung
shap.dependence_plot(("EEG C3-A2_beta_log_energy_entropy", "EEG C4-A1_gamma_power"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C4-A1_gamma_power", "EEG C3-A2_beta_log_energy_entropy"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_wavelet_var_0", "EEG C3-A2_mean"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C4-A1_wavelet_var_0", "EEG C3-A2_sigma_power"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_sigma_power", "EEG C4-A1_wavelet_var_0"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_gamma_log_energy_entropy", "EEG C4-A1_gamma_power"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_beta_log_energy_entropy", "EEG C3-A2_gamma_power"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_gamma_power", "EEG C3-A2_beta_log_energy_entropy"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_theta_log_energy_entropy", "EEG C3-A2_gamma_log_energy_entropy"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_gamma_log_energy_entropy", "EEG C3-A2_theta_log_energy_entropy"), shap_interaction_values, X_test)
shap.dependence_plot(("EEG C3-A2_gamma_svd_entropy", "EEG C3-A2_theta_log_energy_entropy"), shap_interaction_values, X_test)
         

# %%
# Liste der Merkmale, die Frau Hein interessieren
selected_features = [
    "EEG C4-A1_spindle_count",
    "EEG C4-A1_k_complex_count",
    "EEG C3-A2_spindle_count",
    "EEG C3-A2_k_complex_count",
    "EEG A1-A2_spindle_count",
    "EEG A1-A2_k_complex_count",
    "EEG C4-A1_delta_power.1",
    "EEG C3-A2_delta_power.1",
    "EEG A1-A2_delta_power.1"
]

shap_values_selected = shap_values[:, [X_test.columns.get_loc(col) for col in selected_features]]
X_test_selected = X_test[selected_features]

shap.summary_plot(shap_values_selected, X_test_selected)

# %% [markdown]
# **Feature Selection für XGBoost**

# %%
# Ausgabe der verwendeten Features

import pandas as pd

importances = best_xgb_model.feature_importances_

# Erstelle DataFrame zur Analyse der Feature Wichtigkeit
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

zero_importance_features = features_df[features_df['Importance'] == 0]
non_zero_importance_features = features_df[features_df['Importance'] > 0]

# Anzahl der verwendeten und nicht verwendeten Features
num_zero_importance_features = len(zero_importance_features)
num_non_zero_importance_features = len(non_zero_importance_features)

print(f"Anzahl der verwendeten Features: {num_non_zero_importance_features}")
print(f"Anzahl der nicht verwendeten Features: {num_zero_importance_features}")

print("Nicht verwendete Features:")
print(zero_importance_features['Feature'].tolist())


# %%
#XGBoost Feature Selection -> Schlechter als ohne

from sklearn.feature_selection import SelectFromModel

# Erstelle ein Modell mit den besten Parametern
xgb_model = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.03,
    max_depth=4,
    n_estimators=150,
    reg_alpha=3,
    reg_lambda=6,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Feature Selection mit SelectFromModel
selector = SelectFromModel(xgb_model, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"Anzahl der ausgewählten Features: {X_train_selected.shape[1]}")

xgb_model_selected = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.03,
    max_depth=4,
    n_estimators=150,
    reg_alpha=3,
    reg_lambda=6,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model_selected.fit(X_train_selected, y_train)

y_pred_selected = xgb_model_selected.predict(X_test_selected)
y_proba_selected = xgb_model_selected.predict_proba(X_test_selected)[:, 1]

accuracy_selected = accuracy_score(y_test, y_pred_selected)
roc_auc_selected = roc_auc_score(y_test, y_proba_selected)
classification_rep_selected = classification_report(y_test, y_pred_selected)

print("Ergebnisse für das XGBoost-Modell mit ausgewählten Features:")
print(f"Accuracy: {accuracy_selected}")
print(f"ROC-AUC: {roc_auc_selected}")
print(classification_rep_selected)

# %% [markdown]
# **RF**

# %%
#RF

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import shap
import seaborn as sns

features_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/kombinierte_features_SH.csv'
responder_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv'

data = pd.read_csv(features_path)

# Überprüfen, ob 'Patienten ID' als Ganzzahl oder Zeichenfolge formatiert ist und konvertieren
if data['Patienten ID'].dtype != 'object':
    data['Patienten ID'] = data['Patienten ID'].astype(str)

responder_data = pd.read_csv(responder_path)

# Konvertiere 'Patienten ID' in beiden DataFrames in den gleichen Datentyp 
responder_data['Patienten ID'] = responder_data['Patienten ID'].astype(str)

# Zusammenführen der Features und Responder-Labels basierend auf der 'Patienten ID'
combined_df = pd.merge(data, responder_data[['Patienten ID', 'Responder']], on='Patienten ID')

# Überprüfen, ob doppelte Responder-Spalten vorhanden sind, und eine davon entfernen
if 'Responder_x' in combined_df.columns and 'Responder_y' in combined_df.columns:
    combined_df['Responder'] = combined_df['Responder_x']
    combined_df = combined_df.drop(columns=['Responder_x', 'Responder_y'])

# Überprüfen, ob die Spalte 'Responder' nach der Bereinigung vorhanden ist
if 'Responder' not in combined_df.columns:
    raise KeyError("Die 'Responder'-Spalte fehlt nach der Zusammenführung und Bereinigung in 'combined_df'.")

X = combined_df.drop(columns=['Patienten ID', 'ESS_Score', 'Responder'], errors='ignore')
y = combined_df['Responder']

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Berechnung des class_weight basierend auf den Klassenverteilungen
class_weights = {0: len(y_train[y_train == 1]) / len(y_train[y_train == 0]), 1: 1.0}

# Modelltraining unter Berücksichtigung des Klassenungleichgewichts
rf_model = RandomForestClassifier(
    class_weight=class_weights, 
    n_estimators=300,  
    max_depth=8,       
    min_samples_split=20,  
    min_samples_leaf=15,   
    max_features='sqrt',  
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
classification_rep = classification_report(y_test, y_pred)

print("Ergebnisse für das Random Forest-Modell (weiter angepasst):")
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print(classification_rep)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"Spezifität (Specificity): {specificity}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity}")

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad Responder', 'Good Responder'], yticklabels=['Bad Responder', 'Good Responder'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix RF')
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# SHAP-Explainer für Random Forest erstellen
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Struktur überprüfen
print("Struktur der SHAP-Werte:", np.shape(shap_values))

# Extrahiere die SHAP-Werte für die positive Klasse
shap_values_positive_class = shap_values[..., 1]  

# Überprüfe die Dimensionen der extrahierten SHAP-Werte
print("Anzahl der SHAP-Werte für die positive Klasse:", len(shap_values_positive_class[0]))

# Sicherstellen, dass die Dimensionen übereinstimmen
assert len(X_test.columns) == len(shap_values_positive_class[0]), "Die Anzahl der Features stimmt nicht mit den SHAP-Werten überein!"

# Force Ploterstellen
sample_idx = 0  
shap.plots.force(
    explainer.expected_value[1],  
    shap_values_positive_class[sample_idx],
    X_test.iloc[sample_idx]  
)

# %%
#Vergleich Training/Test

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Berechnung und Anzeige der Metriken auf den Testdaten
print("Ergebnisse für das optimierte Random Forest-Modell auf den Testdaten:")
y_pred_test = rf_model.predict(X_test)
y_proba_test = rf_model.predict_proba(X_test)[:, 1]

accuracy_test = accuracy_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_proba_test)
classification_rep_test = classification_report(y_test, y_pred_test)
cm_test = confusion_matrix(y_test, y_pred_test)

tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
specificity_test = tn_test / (tn_test + fp_test)
sensitivity_test = tp_test / (tp_test + fn_test)

print(f"Accuracy: {accuracy_test}")
print(f"ROC-AUC: {roc_auc_test}")
print(f"Spezifität (Specificity): {specificity_test}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity_test}")
print(classification_rep_test)

# Berechnung und Anzeige der Metriken auf den Trainingsdaten
print("\nErgebnisse für das optimierte Random Forest-Modell auf den Trainingsdaten:")
y_pred_train = rf_model.predict(X_train)
y_proba_train = rf_model.predict_proba(X_train)[:, 1]

accuracy_train = accuracy_score(y_train, y_pred_train)
roc_auc_train = roc_auc_score(y_train, y_proba_train)
classification_rep_train = classification_report(y_train, y_pred_train)
cm_train = confusion_matrix(y_train, y_pred_train)

tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
specificity_train = tn_train / (tn_train + fp_train)
sensitivity_train = tp_train / (tp_train + fn_train)

print(f"Accuracy: {accuracy_train}")
print(f"ROC-AUC: {roc_auc_train}")
print(f"Spezifität (Specificity): {specificity_train}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity_train}")
print(classification_rep_train)

# Vergleich der Metriken auf Trainings- und Testdaten
print("\nVergleich der Ergebnisse (Trainingsdaten vs. Testdaten):")
print(f"Accuracy - Training: {accuracy_train}, Test: {accuracy_test}")
print(f"ROC-AUC - Training: {roc_auc_train}, Test: {roc_auc_test}")
print(f"Spezifität - Training: {specificity_train}, Test: {specificity_test}")
print(f"Sensitivität - Training: {sensitivity_train}, Test: {sensitivity_test}")


# %%
# Global Interpretability Plot - Für eine übersicht über die Feature Importance
shap.summary_plot(shap_values_positive_class, X_test, plot_type="bar")

# Summary Plot - Für eine detaillierte Übersicht über die Auswirkungen
shap.summary_plot(shap_values_positive_class, X_test)

# %% [markdown]
# **ANN**

# %%
# ANN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

features_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/kombinierte_features_SH.csv'
responder_path = '/Volumes/FestPLouis/Daten_Schlaflabor_Labels/responder_final.csv'

data = pd.read_csv(features_path)

# Überprüfen, ob 'Patienten ID' als Ganzzahl oder Zeichenfolge formatiert ist und konvertieren
if data['Patienten ID'].dtype != 'object':
    data['Patienten ID'] = data['Patienten ID'].astype(str)

responder_data = pd.read_csv(responder_path)

# Konvertiere 'Patienten ID' in beiden DataFrames in den gleichen Datentyp 
responder_data['Patienten ID'] = responder_data['Patienten ID'].astype(str)

# Zusammenführen der Features und Responder-Labels basierend auf der 'Patienten ID'
combined_df = pd.merge(data, responder_data[['Patienten ID', 'Responder']], on='Patienten ID')

# Überprüfen, ob doppelte Responder-Spalten vorhanden sind, und eine davon entfernen
if 'Responder_x' in combined_df.columns and 'Responder_y' in combined_df.columns:
    combined_df['Responder'] = combined_df['Responder_x']
    combined_df = combined_df.drop(columns=['Responder_x', 'Responder_y'])

X = combined_df.drop(columns=['Patienten ID', 'ESS_Score', 'Responder'], errors='ignore')
y = combined_df['Responder']

# Sicherstellen, dass keine NaNs oder unendlichen Werte in den Daten vorhanden sind
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Anwenden von SMOTE auf die Trainingsdaten, um die Minderheitsklasse zu oversamplen
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train_res.shape[1],)),
    layers.Dropout(0.4),  # Erhöhtes Dropout zur Vermeidung von Overfitting
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # Weitere Schicht hinzugefügt
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Reduzierte Lernrate und Early Stopping
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_proba = model.predict(X_test).flatten()

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
classification_rep = classification_report(y_test, y_pred)

print("Ergebnisse für das ANN-Modell:")
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print(classification_rep)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"Spezifität (Specificity): {specificity}")
print(f"Sensitivität (Sensitivity/Recall): {sensitivity}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Learning Curve (ANN)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad Responder', 'Good Responder'], yticklabels=['Bad Responder', 'Good Responder'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix ANN')
plt.show()

# Berechne Features Importance 
def scoring_func(estimator, X, y):
    y_pred_proba = estimator.predict(X)
    return roc_auc_score(y, y_pred_proba)

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=scoring_func)

importances = result.importances_mean
indices = np.argsort(importances)[::-1]
feature_names = X.columns

print("Top Features nach Wichtigkeit sortiert (Permutation Importance):")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]}")

plt.figure(figsize=(12, 8))
plt.title("Feature Importances (Permutation Importance)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# Erstelle einen SHAP-Explainer für das neuronale Netz
explainer = shap.DeepExplainer(model, X_train[:100])

# Berechne SHAP-Werte für den Testdatensatz
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

# Berechne die mittleren absoluten SHAP-Interaktionswerte
interaction_values = np.dot(shap_values.T, shap_values) / shap_values.shape[0]

# Erstelle eine Heatmap der SHAP-Interaktionswerte
plt.figure(figsize=(12, 10))
sns.heatmap(interaction_values, cmap='coolwarm', xticklabels=X.columns, yticklabels=X.columns)
plt.title("SHAP Feature Interaction Values Heatmap")
plt.show()

# %%
# Global Interpretability Plot - Als Übersicht über die Feature Importance 
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# Global Interpretability Plot - Für eine detaillierte Übersicht über die Auswirkungen
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# %%
#Feature Interaktionen

import numpy as np
import pandas as pd

interaction_importance_df = pd.DataFrame(interaction_values, index=X.columns, columns=X.columns)

np.fill_diagonal(interaction_importance_df.values, 0)

# Finde die stärksten Interaktionen 
N = 11  
strongest_interactions = interaction_importance_df.unstack().sort_values(ascending=False).drop_duplicates().head(N)

print("Stärkste Feature-Interaktionen:")
print(strongest_interactions)

# %%
# Speichere die Spaltennamen, bevor du die Daten transformierst
feature_names = X.columns

# %%
# SHAP Dependence Plot für die stärksten Interaktionen
shap.dependence_plot("EEG C4-A1_svd_entropy", shap_values, X_test, interaction_index="EEG C4-A1_hjorth_mobility", feature_names=feature_names)
shap.dependence_plot("EEG C4-A1_beta_log_energy_entropy", shap_values, X_test, interaction_index="EEG C4-A1_sigma_log_energy_entropy", feature_names=feature_names)
shap.dependence_plot("EEG C4-A1_alpha_log_energy_entropy", shap_values, X_test, interaction_index="EEG C4-A1_log_energy_entropy", feature_names=feature_names)

# %%
# SHAP 3D-Scatter Plot gür die stärkste Interaktion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, feature_names.get_loc('EEG C4-A1_svd_entropy')],
           X_test[:, feature_names.get_loc('EEG C4-A1_hjorth_mobility')],
           shap_values[:, feature_names.get_loc('EEG C4-A1_svd_entropy')],
           c=shap_values[:, feature_names.get_loc('EEG C4-A1_svd_entropy')], cmap='coolwarm')

ax.set_xlabel('EEG C4-A1_svd_entropy')
ax.set_ylabel('EEG C4-A1_hjorth_mobility')
ax.set_zlabel('SHAP value for EEG C4-A1_svd_entropy')

plt.show()

# %% [markdown]
# **Berechnen des RDIs**

# %%
#Berechnen RDI anhand der RML Informationen

# Momentan nicht verwendet

import os
import xml.etree.ElementTree as ET
import pandas as pd

def load_rml_file(file_path):
    #Lädt die .rml-Datei und gibt das Wurzelelement zurück.
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Fehler beim Parsen der Datei {file_path}: {e}")
        return None

def extract_sleep_stages(root, namespace):
    #Extrahiert die Schlafstadien und deren Startzeiten aus der .rml-Datei.
    sleep_stages = []
    for elem in root.iter(f'{namespace}Stage'):
        if 'Type' in elem.attrib and 'Start' in elem.attrib:
            stage_type = elem.attrib['Type']
            start_time = int(elem.attrib['Start'])
            sleep_stages.append((stage_type, start_time))
    sleep_stages.sort(key=lambda x: x[1])
    return sleep_stages

def calculate_total_sleep_time_from_stages(root, namespace):
    #Berechnet die Gesamtschlafzeit basierend auf den Startzeiten der Schlafstadien.
    start_times = []
    for elem in root.iter(f'{namespace}Stage'):
        if 'Start' in elem.attrib:
            start_time = float(elem.attrib['Start'])
            start_times.append(start_time)
    if start_times:
        total_sleep_time_hours = (max(start_times) - min(start_times)) / 3600
    else:
        total_sleep_time_hours = 0
    return total_sleep_time_hours

def extract_events(root, namespace):
    #Zählt alle Apnoen, Hypopnoen, RERAs und andere relevante Ereignisse in der .rml-Datei.
    apnoe_count = 0
    hypopnoe_count = 0
    rera_count = 0  

    for event in root.iter(f'{namespace}Event'):
        event_family = event.attrib.get('Family', '').lower()
        event_type = event.attrib.get('Type', '').lower()

        if event_family == 'respiratory':
            if event_type in ['centralapnea', 'obstructiveapnea', 'mixedapnea']:
                apnoe_count += 1
            elif event_type == 'hypopnea':
                hypopnoe_count += 1
            elif event_type == 'rera':
                rera_count += 1

    return apnoe_count, hypopnoe_count, rera_count

def calculate_rdi(apnoe_count, hypopnoe_count, rera_count, total_sleep_time_hours):
    #Berechnet den Respiratory Disturbance Index (RDI) unter Einbeziehung zusätzlicher Ereignisse.
    if total_sleep_time_hours > 0:
        total_events = apnoe_count + hypopnoe_count + rera_count
        rdi = total_events / total_sleep_time_hours
    else:
        rdi = 0
    return rdi

def process_rml_files(relevant_folders_df):
    #Durchläuft die relevanten Verzeichnisse und verarbeitet alle .rml-Dateien.
    results = []
    for index, row in relevant_folders_df.iterrows():
        folder = row['Folder'] 
        patient_id = row['PatientID']  

        for root_dir, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.rml'):
                    file_path = os.path.join(root_dir, file)
                    
                    print(f"Processing file for Patient ID: {patient_id}")

                    root = load_rml_file(file_path)
                    if root is None:
                        continue

                    namespace = '{http://www.respironics.com/PatientStudy.xsd}'

                    total_sleep_time_hours = calculate_total_sleep_time_from_stages(root, namespace)
                    apnoe_count, hypopnoe_count, rera_count = extract_events(root, namespace)
                    rdi = calculate_rdi(apnoe_count, hypopnoe_count, rera_count, total_sleep_time_hours)

                    results.append({
                        'patient_id': patient_id,
                        'total_sleep_time_hours': total_sleep_time_hours,
                        'apnoe_count': apnoe_count,
                        'hypopnoe_count': hypopnoe_count,
                        'rera_count': rera_count,
                        'rdi': rdi
                    })
                    
                    print(f"Results for Patient ID {patient_id}:")
                    print(f"  Gesamtschlafzeit: {total_sleep_time_hours} Stunden")
                    print(f"  Apnoen: {apnoe_count}")
                    print(f"  Hypopnoen: {hypopnoe_count}")
                    print(f"  RERAs: {rera_count}")
                    print(f"  RDI: {rdi}")

    return results

# Lade die relevanten Verzeichnisnamen aus der CSV
relevant_folders_df = pd.read_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/relevant_folders.csv')
print(relevant_folders_df.columns)

# Verarbeite die Dateien in den relevanten Verzeichnissen
results = process_rml_files(relevant_folders_df)

# Optional: Ergebnisse speichern
df = pd.DataFrame(results)
df.to_csv('/Volumes/FestPLouis/Daten_Schlaflabor_Labels/rdi_patients.csv', index=False)


