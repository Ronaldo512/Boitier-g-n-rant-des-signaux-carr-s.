"""
Le 26 Janvier 2025 à 1h 35min.

   Autor: Ronaldo Yansunnu ADJOVI 
   
"""
import os
import tarfile
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram, butter, filtfilt, iirnotch, detrend

# Configuration
teager_kaiser = 1
unzip_tar = 1

#Direction vers le dossier du fichier
rep = 'C:/Users/hp EliteBook 840 G3/Desktop/Stage/python/Transformer/tryyy'

# Liste des fichiers dans le repertoire
D = os.listdir(rep)
filename_all = [f for f in D if ".tar.gz" in f]
filename = [{'track': None} for _ in range(len(filename_all))]

if unzip_tar == 1:
    for i, file in enumerate(filename_all):
        filename[i]['track'] = file
        try:
            with tarfile.open(os.path.join(rep, file), 'r:gz') as tar:
                tar.extractall(os.path.join(rep, file[:-7]))
        except Exception as e:
            warnings.warn(f'Cannot read the YAML file {file}: {e}')
else:
    print('Not untar')
    for i, file in enumerate(filename_all):
        filename[i]['track'] = file

# Lecture des fichiers waveforms.bin et metadata.yaml
Data = []
yaml_data = []
Fs = []
startime_val = []
trigger = []
end_time_temp = []
for i, file in enumerate(filename):
    data_receive = os.path.join(rep, file['track'][:-7], 'waveforms.bin')
    with open(data_receive, 'rb') as f:
        Data.append({'track': np.fromfile(f, dtype=np.float32)})
    
    metadata_file = os.path.join(rep, file['track'][:-7], 'metadata.yaml')
    with open(metadata_file, 'r') as f:
        metadata_temp = f.readlines()
        yaml_data.append({'data': metadata_temp})
    
    metadata = yaml_data[i]['data']
    startime_temp = [line for line in metadata if 'StartTime' in line][0]
    Trigger_temp = [line for line in metadata if 'Timestamp' in line]
    Periode = [line for line in metadata if 'Period' in line][0]
    Fs.append({'Fs': 1 / float(Periode.strip().split()[-1])})
    
    startime_val.append(float(startime_temp.split()[1]))
    trigger.append({'val': [float(ts.split()[1]) for ts in Trigger_temp]})
    end_time_temp.append(startime_val[i] + len(Data[i]['track']) / 3 / Fs[i]['Fs'])

# Conversion des timestamps en datetime
start_time = [datetime.fromtimestamp(ts) for ts in startime_val]
end_time = [datetime.fromtimestamp(ts) for ts in end_time_temp]
horloge = []
for i in range(len(start_time)):
    timestamps = np.linspace(
        start_time[i].timestamp(), 
        end_time[i].timestamp(), 
        len(Data[i]['track']) // 3
    )
    datetime_values = [datetime.utcfromtimestamp(ts) for ts in timestamps]
    horloge.append({'val': datetime_values})

# Séparation des données
VCP, ICP, REF = [], [], []
for i in range(len(Data)):
    vcp_data = Data[i]['track'][:len(Data[i]['track']) // 3]
    icp_data = Data[i]['track'][len(Data[i]['track']) // 3:2 * len(Data[i]['track']) // 3]
    ref_data = Data[i]['track'][2 * len(Data[i]['track']) // 3:]

    # Application des filtres Notch
    Fnotch = 50  # Fréquence de coupure (Hz)
    BW = 2  # Largeur de bande du Notch
    Fs_value = Fs[0]['Fs']  # Fréquence d'échantillonnage
    b, a = iirnotch(Fnotch, Fnotch / BW, Fs_value)

    # Appliquer les filtres Notch à VCP, ICP et REF
    VCP.append({'val': vcp_data, 'notch': filtfilt(b, a, vcp_data)})
    ICP.append({'val': icp_data, 'notch': filtfilt(b, a, icp_data)})
    REF.append({'val': ref_data, 'notch': filtfilt(b, a, ref_data)})

# Filtrage passe-bande
ftype = 'band'
Wn = [0.5, 3]
order = 2
b_band, a_band = butter(order, np.array(Wn) * 2 / Fs_value, btype=ftype)

for i in range(len(Data)):
    ICP[i]['fit'] = filtfilt(b_band, a_band, ICP[i]['notch'])
    REF[i]['fit'] = filtfilt(b_band, a_band, REF[i]['notch'])
    VCP[i]['fit'] = filtfilt(b_band, a_band, VCP[i]['notch'])

# Application du filtrage passe-bas
order = 5
Wn = 5  # Fréquence de coupure en Hz
b_low, a_low = butter(order, Wn * 2 / Fs[i]['Fs'], btype='low')

for i in range(len(REF)):
    ICP[i]['fit2'] = filtfilt(b_low, a_low, ICP[i]['notch'])
    REF[i]['fit2'] = filtfilt(b_low, a_low, REF[i]['notch'])
    VCP[i]['fit2'] = filtfilt(b_low, a_low, VCP[i]['notch'])

# Détection Teager-Kaiser
if teager_kaiser == 1:
    for i in range(len(REF)):
        REF[i]['TK'] = np.zeros(len(REF[i]['fit']))
        REF[i]['TK'][1:-1] = np.power(REF[i]['fit'][1:-1], 2) - REF[i]['fit'][:-2] * REF[i]['fit'][2:]
        REF[i]['TK_abs'] = np.abs(REF[i]['TK'])

    for i in range(len(REF)):
        T = 1 / Fs[i]['Fs']
        REF[i]['time'] = np.arange(0, len(REF[i]['TK_abs'])) * T

    # Visualisation
    try:
        plt.close('Check_signal_filtered2')
    except:
        pass
    plt.figure('Check_signal_filtered2')
    plt.subplot(212)
    ax2 = plt.gca()
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.set_title('Teager-Kaiser')

    for i in range(len(REF)):
        plt.plot(horloge[i]['val'], REF[i]['TK_abs'], 'r', linewidth=1)

    plt.tight_layout()
    plt.show()
    
    # Tracer les signaux filtrés et non filtrés
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i in range(len(horloge)):
    # VCP Signal
    axs[0].plot(horloge[i]['val'], VCP[i]['val'], 'b', label='Sans filtre' if i == 0 else "")
    axs[0].plot(horloge[i]['val'], VCP[i]['notch'], 'm', label='Avec filtre Notch' if i == 0 else "")
    axs[0].plot(horloge[i]['val'], VCP[i]['fit'], 'r', label='Avec filtre Passe-bande' if i == 0 else "")
    axs[0].plot(horloge[i]['val'], VCP[i]['fit2'], 'g', label='Avec filtre Passe-bas' if i == 0 else "")

    axs[0].set_title('VCP Signal')

    # ICP Signal
    axs[1].plot(horloge[i]['val'], ICP[i]['val'], 'g', label='Sans filtre' if i == 0 else "")
    axs[1].plot(horloge[i]['val'], ICP[i]['notch'], 'm', label='Avec filtre Notch' if i == 0 else "")
    axs[1].plot(horloge[i]['val'], ICP[i]['fit'], 'y', label='Avec filtre Passe-bande' if i == 0 else "")
    axs[1].plot(horloge[i]['val'], ICP[i]['fit2'], 'r', label='Avec filtre Passe-bas' if i == 0 else "")
    axs[1].set_title('ICP Signal')

    # REF Signal
    axs[2].plot(horloge[i]['val'], REF[i]['val'], 'k', label='Sans filtre' if i == 0 else "")
    axs[2].plot(horloge[i]['val'], REF[i]['notch'], 'orange', label='Avec filtre Notch' if i == 0 else "")
    axs[2].plot(horloge[i]['val'], REF[i]['fit'], 'purple', label='Avec filtre Passe-bande' if i == 0 else "")
    axs[2].plot(horloge[i]['val'], REF[i]['fit2'], 'red', label='Avec filtre Passe-bas' if i == 0 else "")
    axs[2].set_title('REF Signal')
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


# Vérification et tracé des déclencheurs
for i in range(len(trigger)):
    for j in range(len(trigger[i]['val'])):
        try:
            trigger_time = datetime.utcfromtimestamp(trigger[i]['val'][j])
            min_y = min(REF[i]['fit2'])
            max_y = max(REF[i]['fit2'])
            # Tracer la ligne verticale
            plt.plot([trigger_time, trigger_time], [min_y, max_y], 'k:', linewidth=2)
        except Exception as e:
            print(f"Erreur pour Trigger[{i}]['val'][{j}]: {e}")

# Détection des ondes avec des seuils
def detection(magfield, seuil_bas, seuil_haut, persistance, intervalle, Fs):
    """Détecte les signaux dépassant un seuil dans une bande de fréquences."""
    actif = np.zeros(len(magfield), dtype=int)

    # Détection des seuils hauts et bas
    high = magfield > seuil_haut
    high_beginning = np.where(np.diff(high.astype(int)) == 1)[0] + 1
    high_end = np.where(np.diff(high.astype(int)) == -1)[0] + 1

    low = magfield > seuil_bas
    low_beginning = np.where(np.diff(low.astype(int)) == 1)[0] + 1
    low_end = np.where(np.diff(low.astype(int)) == -1)[0] + 1

    # Ajuster les longueurs si necessaire
    if len(low_beginning) > len(low_end):
        low_beginning = low_beginning[:-1]
    if len(high_beginning) > len(high_end):
        high_beginning = high_beginning[:-1]

    # Synchroniser les indices
    try:
        i = 0
        while low_beginning[0] > low_end[i]:
            i += 1
    except IndexError:
        pass

    if i > 0:
        high_beginning = high_beginning[:-i]
        low_beginning = low_beginning[:-i]
        low_end = low_end[i:]
        high_end = high_end[i:]

    print(f"Number of detections = {len(low_beginning)}")

    for i in range(len(low_beginning)):
        for j in range(len(high_beginning)):
            if high_end[j] - high_beginning[j] >= persistance * Fs:
                if low_beginning[i] <= high_beginning[j] and high_end[j] <= low_end[i]:
                    index = 1
                    while i + index < len(low_end):
                        if low_beginning[i + index] - low_end[i] >= intervalle * Fs:
                            actif[low_beginning[i]:low_end[i + index - 1]] = 1
                            break
                        index += 1
                        if i + index >= len(low_end):
                            actif[low_beginning[i]:low_end[i]] = 1

    detect = np.diff(actif, prepend=0)
    DB = np.where(detect == 1)[0]
    FB = np.where(detect == -1)[0]

    # Ajustement des longueurs des DB et FB
    if len(DB) > len(FB):
        DB = DB[:len(FB)]
    elif len(DB) < len(FB):
        FB = FB[:len(DB)]

    return actif, DB, FB


# Application de la détection sur les données
detect = []
for i in range(len(REF)):
    seuil_bas = 0.3 * np.std(REF[i]["TK_abs"])
    seuil_haut = 0.5 * np.std(REF[i]["TK_abs"])
    persistance = 0.5
    intervalle = 0

    actif, DB, FB = detection(
        REF[i]["TK_abs"],
        seuil_bas,
        seuil_haut,
        persistance,
        intervalle,
        Fs[i]["Fs"]
    )

    detect.append({
        "seuil_bas": seuil_bas,
        "seuil_haut": seuil_haut,
        "persistance": persistance,
        "intervalle": intervalle,
        "actif": actif,
        "DBok": DB,
        "FBok": FB,
    })


# Créer une figure unique pour afficher toutes les courbes
plt.figure(figsize=(14, 8))

# Couleurs pour les détections
detection_color = 'r'
star_color = 'k'

# Tracer chaque signal sur la mÃ©me figure
for i in range(len(detect)):
    # Tracer le signal brut
    plt.plot(horloge[i]["val"], REF[i]["TK_abs"], label=f"Signal {i+1}", linewidth=1.5)
    
    # Tracer les seuils
    plt.plot(horloge[i]["val"], detect[i]["seuil_bas"] * np.ones(len(REF[i]["TK_abs"])), '--', color='b', label=f"Seuil bas {i+1}" if i == 0 else "")
    plt.plot(horloge[i]["val"], detect[i]["seuil_haut"] * np.ones(len(REF[i]["TK_abs"])), '--', color='g', label=f"Seuil haut {i+1}" if i == 0 else "")
    
    # Tracer les dÃ©tections
    plt.plot(horloge[i]["val"], detect[i]["actif"] * detect[i]["seuil_haut"], color=detection_color, label=f"DÃƒÂ©tection {i+1}" if i == 0 else "", linewidth=2)

    # Ajouter les points dÃ©tectÃ©s au sommet des troncatures
    for j in range(len(detect[i]["DBok"])):
        # Intervalle de dÃ©tection
        start_idx = detect[i]["DBok"][j]
        end_idx = detect[i]["FBok"][j]

        # Identifier le sommet (maximum) dans cet intervalle
        detection_range = REF[i]["TK_abs"][start_idx:end_idx]
        max_value = np.max(detection_range)
        max_idx = np.argmax(detection_range) + start_idx

        # Temps correspondant au sommet
        time_at_max = horloge[i]["val"][max_idx]

        # Ajouter une étoile au sommet
        plt.plot(time_at_max, max_value, marker='*', color=star_color, markersize=10)

# Ajouter des détails au graphique
plt.title("Détection des signaux", fontsize=16)
plt.xlabel("Temps", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.legend()
#plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

#plt.close('all')
# Configuration des paramètres pour le spectrogramme
Fs_value = Fs[0]['Fs']
fs=Fs_value
movingwin = [4, 1]  # Fenètre glissante : longueur et pas

# Fonction personnalisée pour réaliser un spectrogramme multitaper
def mpecgram(signal, fs, movingwin, params):
    """Génère un spectrogramme et retourne les résultats."""
    win_len, win_step = movingwin
    nperseg = int(win_len * fs)
    noverlap = int((win_len - win_step) * fs)
    fmin, fmax = params['fpass']
    
    f, t, Sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density'
    )
    
    
    # Filtrer la bande de fréquences spÃ©cifiée
    freq_idx = (f >= fmin) & (f <= fmax)
    return Sxx[freq_idx, :], t, f[freq_idx]

#Concatenation des signaux une fois que le detrend est appliquÃ©
raw_signal_combined = np.concatenate([detrend(REF[i]['val'], type='linear') for i in range(len(REF))])
notch_signal_combined = np.concatenate([detrend(REF[i]['notch'], type='linear') for i in range(len(REF))])

params = {
    'Fs': Fs[0]['Fs'],  # Fréquence d'échantillonnage
    'fpass': [0.1, 500]  # Bande de fréquences d'interét
}
# Calcul du spectrogramme multitaper
Ptotal, TT, freqdom = mpecgram(raw_signal_combined, Fs_value, movingwin, params)
Ptotal_notch, _, _ = mpecgram(notch_signal_combined, Fs_value, movingwin, params)

# A‰éviter log(0)
Ptotal += np.finfo(float).eps  
Ptotal_notch += np.finfo(float).eps

# Moyenne des spectrogrammes
Ptotal_mean = np.mean(Ptotal, axis=1)
Ptotal_notch_mean = np.mean(Ptotal_notch, axis=1)

# Normalisation pour éviter un décalage visuel
Ptotal_mean /= np.max(Ptotal_mean)
Ptotal_notch_mean /= np.max(Ptotal_notch_mean)

# Tracer la Transformée de Fourier en dB
fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
ax1.plot(freqdom, 20 * np.log10(Ptotal_mean), label="Signal brut")
ax1.plot(freqdom, 20 * np.log10(Ptotal_notch_mean), label="Signal avec filtre Notch", linestyle='--')
plt.xlim(0.1, 160)
ax1.set_title("Transformée de Fourier")
ax1.set_xlabel("Fréquence (Hz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.legend()
ax1.grid()

# Spectrogrammes
ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=1) 
pcm1 = ax2.pcolormesh(TT, freqdom, 20 * np.log10(Ptotal), cmap='nipy_spectral', shading='auto')
ax2.set_title("Raw")
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel("Fréquence (Hz)")
plt.ylim(0.1 ,160)
fig.colorbar(pcm1, ax=ax2, label='Power (dB)')

ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=1) 
pcm2 = ax3.pcolormesh(TT, freqdom, 20 * np.log10(Ptotal), cmap='nipy_spectral', shading='auto')
ax3.set_title("Notch")
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Fréquence (Hz)")
plt.ylim(0.1 ,160)
fig.colorbar(pcm2, ax=ax3, label='Power (dB)')

ax4 = plt.subplot2grid((2, 3), (1, 1), rowspan=1) 
pcm3 = ax4.pcolormesh(TT, freqdom, 20 * np.log10(Ptotal), cmap='nipy_spectral', shading='auto')
ax4.set_title("Raw")
ax4.set_xlabel("Temps (s)")
ax4.set_ylabel("Fréquence (Hz)")
plt.ylim(0.1 ,60)
fig.colorbar(pcm3, ax=ax4, label='Power (dB)')

ax5 = plt.subplot2grid((2, 3), (1, 2), rowspan=1) 
pcm4 = ax5.pcolormesh(TT, freqdom, 20 * np.log10(Ptotal), cmap='nipy_spectral', shading='auto')
ax5.set_title("Notch")
ax5.set_xlabel("Temps (s)")
ax5.set_ylabel("Fréquence (Hz)")
plt.ylim(0.1 ,60)
fig.colorbar(pcm4, ax=ax5, label='Power (dB)')

#plt.tight_layout()
#plt.show()

