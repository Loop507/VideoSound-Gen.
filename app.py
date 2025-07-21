import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional, Union
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa
import librosa.display
import time
from scipy.ndimage import uniform_filter1d

# --- Costanti ---
AUDIO_SAMPLE_RATE = 44100  # Hz
AUDIO_FPS = 25 # Frame rate virtuale per l'audio, se non c'è video
MAX_DURATION = 180 # Secondi, durata massima audio/video per evitare carichi eccessivi
FFMPEG_PATH = "ffmpeg" # Assicurati che ffmpeg sia nel PATH o specifica il percorso completo

# Risoluzioni comuni per l'output video
FORMAT_RESOLUTIONS = {
    "Originale": None, # Manterrà la risoluzione originale del video caricato
    "1080p (Full HD)": (1920, 1080),
    "720p (HD)": (1280, 720),
    "480p (SD)": (854, 480)
}

# --- Funzioni Helper ---
def check_ffmpeg():
    """Controlla se FFmpeg è installato e accessibile."""
    try:
        subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def _map_value(
    value: Union[float, np.ndarray], # Accetta sia float che ndarray
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float
) -> Union[float, np.ndarray]:
    """Mappa un valore o un array di valori da un intervallo a un altro, con clipping."""
    if old_max == old_min:
        if isinstance(value, np.ndarray):
            return np.full_like(value, new_min, dtype=np.float32)
        else:
            return new_min # Evita divisione per zero se l'intervallo è piatto
    
    # Clipa il valore all'interno del vecchio intervallo prima di mapparlo
    clipped_value = np.clip(value, old_min, old_max)
    
    mapped_value = new_min + (clipped_value - old_min) * (new_max - new_min) / (old_max - old_min)
    
    # Se l'input era un array, ritorna un array. Altrimenti, un float.
    if isinstance(value, np.ndarray):
        return mapped_value.astype(np.float32)
    else:
        return float(mapped_value) # Converti a float nativo per compatibilità

def validate_video_file(uploaded_file):
    """Controlla la dimensione del file video e il tipo."""
    if uploaded_file.size > 200 * 1024 * 1024: # 200 MB limite
        st.error("❌ Il file video è troppo grande. Dimensione massima consentita: 200 MB.")
        return False
    if not uploaded_file.type.startswith("video/"):
        st.error("❌ Il file caricato non è un video valido.")
        return False
    return True

def analyze_video_frames(video_path: str, progress_bar, status_text) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], int, int, float, float
]:
    """
    Analizza i frame di un video per estrarre luminosità, dettaglio, movimento,
    variazione del movimento e centro orizzontale.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Impossibile aprire il video: {video_path}")
        return None, None, None, None, None, 0, 0, 0.0, 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    if frame_count == 0 or fps == 0:
        st.error("❌ Il video non contiene frame o ha un FPS pari a zero. Potrebbe essere corrotto.")
        cap.release()
        return None, None, None, None, None, 0, 0, 0.0, 0.0

    brightness_values = []
    detail_values = []
    movement_values = []
    horizontal_center_values = []
    prev_frame_gray = None

    start_time = time.time()
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Luminosità (media dei pixel in scala di grigi)
        brightness_values.append(np.mean(gray_frame) / 255.0)

        # Dettaglio/Contrasto (varienza di Laplacian)
        # La varianza del Laplaciano è un buon indicatore di nitidezza/dettaglio
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        detail_values.append(laplacian_var)

        # Movimento (differenza assoluta tra frame)
        if prev_frame_gray is not None:
            diff_frame = cv2.absdiff(gray_frame, prev_frame_gray)
            movement_values.append(np.mean(diff_frame) / 255.0)
        else:
            movement_values.append(0.0) # Primo frame non ha movimento
        prev_frame_gray = gray_frame

        # Centro orizzontale di "attività" (basato sull'intensità orizzontale)
        # Calcola la media delle intensità dei pixel per ogni colonna
        column_means = np.mean(gray_frame, axis=0)
        # Calcola il "centro di massa" delle intensità delle colonne
        if np.sum(column_means) > 0:
            center_x = np.sum(np.arange(width) * column_means) / np.sum(column_means)
            horizontal_center_values.append(center_x / width) # Normalizza tra 0 e 1
        else:
            horizontal_center_values.append(0.5) # Centro se il frame è nero

        # Aggiorna la progress bar ogni 50 frame o alla fine
        if i % 50 == 0 or i == frame_count - 1:
            progress = (i + 1) / frame_count
            progress_bar.progress(progress)
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time
            
            status_text.text(f"Analisi: {progress*100:.1f}% completata. "
                             f"Tempo stimato rimanente: {remaining_time:.1f}s")
            
    cap.release()

    # Normalizza i valori di dettaglio e movimento
    brightness_data = np.array(brightness_values, dtype=np.float32)
    detail_data = np.array(detail_values, dtype=np.float32)
    movement_data = np.array(movement_values, dtype=np.float32)
    horizontal_center_data = np.array(horizontal_center_values, dtype=np.float32)

    # Normalizzazione per Dettaglio (potrebbe essere necessario un min/max dinamico)
    if detail_data.max() > detail_data.min():
        detail_data = (detail_data - detail_data.min()) / (detail_data.max() - detail_data.min())
    else:
        detail_data = np.zeros_like(detail_data) # Tutti uguali, imposta a zero

    # Normalizzazione per Movimento
    if movement_data.max() > movement_data.min():
        movement_data = (movement_data - movement_data.min()) / (movement_data.max() - movement_data.min())
    else:
        movement_data = np.zeros_like(movement_data)

    # Calcola la variazione del movimento (per il glitch)
    variation_movement_data = np.abs(np.diff(movement_data, prepend=movement_data[0]))
    # Normalizza la variazione del movimento
    if variation_movement_data.max() > variation_movement_data.min():
        variation_movement_data = (variation_movement_data - variation_movement_data.min()) / \
                                  (variation_movement_data.max() - variation_movement_data.min())
    else:
        variation_movement_data = np.zeros_like(variation_movement_data)

    # Clip finale a 0-1 per tutti i dati per sicurezza
    brightness_data = np.clip(brightness_data, 0.0, 1.0)
    detail_data = np.clip(detail_data, 0.0, 1.0)
    movement_data = np.clip(movement_data, 0.0, 1.0)
    variation_movement_data = np.clip(variation_movement_data, 0.0, 1.0)
    horizontal_center_data = np.clip(horizontal_center_data, 0.0, 1.0)

    # DEBUG: Controlla che i dati siano finiti dopo l'analisi
    if not np.all(np.isfinite(brightness_data)):
        st.error("❌ brightness_data contiene valori non finiti dopo l'analisi.")
        brightness_data[~np.isfinite(brightness_data)] = 0.0
    if not np.all(np.isfinite(detail_data)):
        st.error("❌ detail_data contiene valori non finiti dopo l'analisi.")
        detail_data[~np.isfinite(detail_data)] = 0.0
    if not np.all(np.isfinite(movement_data)):
        st.error("❌ movement_data contiene valori non finiti dopo l'analisi.")
        movement_data[~np.isfinite(movement_data)] = 0.0
    if not np.all(np.isfinite(variation_movement_data)):
        st.error("❌ variation_movement_data contiene valori non finiti dopo l'analisi.")
        variation_movement_data[~np.isfinite(variation_movement_data)] = 0.0
    if not np.all(np.isfinite(horizontal_center_data)):
        st.error("❌ horizontal_center_data contiene valori non finiti dopo l'analisi.")
        horizontal_center_data[~np.isfinite(horizontal_center_data)] = 0.5 # Metti a centro

    return brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, duration

# --- Classe AudioGenerator ---
class AudioGenerator:
    def __init__(self, sample_rate: int, fps: int):
        self.sample_rate = sample_rate
        self.fps = fps
        self.frame_to_sample_ratio = sample_rate / fps

    def _map_value_to_samples(self, control_data: np.ndarray, current_frame: int, min_val: float, max_val: float) -> float:
        """Mappa un valore di controllo basato sul frame corrente a un intervallo specificato."""
        if current_frame >= len(control_data):
            current_frame = len(control_data) - 1 # Evita IndexError per il padding di librosa
        mapped_val = _map_value(control_data[current_frame], 0.0, 1.0, min_val, max_val)
        return mapped_val

    def generate_subtractive_waveform(self, num_samples: int, waveform_type: str = "sawtooth") -> np.ndarray:
        """Genera una forma d'onda base."""
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False, dtype=np.float32)
        
        # Frequenza base fissa, puoi renderla modulabile dalla luminosità se vuoi
        base_freq = 440 # Hz

        if waveform_type == "sine":
            waveform = np.sin(2 * np.pi * base_freq * t)
        elif waveform_type == "square":
            waveform = np.sign(np.sin(2 * np.pi * base_freq * t))
        elif waveform_type == "sawtooth":
            waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
        elif waveform_type == "triangle":
            waveform = 2 * np.abs(2 * (t * base_freq - np.floor(t * base_freq + 0.5))) - 1
        else:
            waveform = np.zeros(num_samples, dtype=np.float32) # Default a silenzio se tipo sconosciuto
        
        return waveform
    
    def apply_dynamic_filter(self,
                             audio_data: np.ndarray,
                             brightness_data: np.ndarray,
                             detail_data: np.ndarray,
                             min_cutoff: float,
                             max_cutoff: float,
                             min_res: float,
                             max_res: float,
                             progress_bar,
                             status_text,
                             filter_type: str = "lowpass") -> np.ndarray:
        """
        Applica un filtro dinamico (lowpass, highpass, bandpass, bandstop) all'audio.
        La frequenza di taglio è modulata dalla luminosità, la risonanza dal dettaglio.
        """
        filtered_audio = np.zeros_like(audio_data, dtype=np.float32)
        
        # Interpolazione dei dati di controllo per l'audio
        num_frames = len(brightness_data)
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, len(audio_data))

        brightness_interp = np.interp(audio_indices, frames_indices, brightness_data)
        detail_interp = np.interp(audio_indices, frames_indices, detail_data)
        
        # Clipa i valori interpolati per sicurezza
        brightness_interp = np.clip(brightness_interp, 0.0, 1.0)
        detail_interp = np.clip(detail_interp, 0.0, 1.0)

        # DEBUG: Controlla che gli interpolati siano finiti
        if not np.all(np.isfinite(brightness_interp)):
            st.warning("⚠️ brightness_interp contiene NaN/inf nel filtro. Sostituzione con 0.5.")
            brightness_interp[~np.isfinite(brightness_interp)] = 0.5
        if not np.all(np.isfinite(detail_interp)):
            st.warning("⚠️ detail_interp contiene NaN/inf nel filtro. Sostituzione con 0.5.")
            detail_interp[~np.isfinite(detail_interp)] = 0.5

        # Applica una media mobile sui dati interpolati per smussare le transizioni
        window_size = int(self.sample_rate / self.fps / 2) # Mezza finestra basata su mezzo frame
        if window_size % 2 == 0: window_size += 1 # Assicura che sia dispari per uniform_filter1d
        if window_size > 1:
            brightness_interp = uniform_filter1d(brightness_interp, size=window_size)
            detail_interp = uniform_filter1d(detail_interp, size=window_size)

        for i in range(len(audio_data)):
            
            # Calcola cutoff e risonanza per il sample corrente
            # La luminosità controlla la frequenza di taglio (da min_cutoff a max_cutoff)
            current_cutoff = _map_value(brightness_interp[i], 0.0, 1.0, min_cutoff, max_cutoff)
            
            # Il dettaglio controlla la risonanza (da min_res a max_res)
            current_resonance = _map_value(detail_interp[i], 0.0, 1.0, min_res, max_res)

            # Normalizza la frequenza di taglio per butter
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = current_cutoff / nyquist

            # Limita la frequenza di taglio normalizzata per evitare problemi
            if normal_cutoff >= 1.0:
                normal_cutoff = 0.99
            elif normal_cutoff <= 0.0:
                normal_cutoff = 0.01 # Evita 0 per logaritmico, e evita frequenze non valide
            
            # Calcola Wn (frequenza di taglio normalizzata) e Q (risonanza) per il filtro
            # Per butter, Wn è la frequenza normalizzata, Q è la risonanza (non direttamente mappabile 1:1 con Q del filtro risonante)
            # Per un filtro passa-banda/stop, Wn può essere una tupla (low, high)
            
            # Utilizziamo un filtro IIR a stato variabile per un controllo più diretto di Q
            # Tuttavia, scrivi un filtro a stato variabile è complesso.
            # Per semplicità, useremo `librosa.effects.band_biquad` o filtri Butterworth
            # `librosa` non ha filtri dinamici per Q, quindi simuliamo con Butterworth o lo escludiamo.
            
            # Per questo esempio, usiamo Butterworth con una frequenza di taglio variabile.
            # La risonanza Q non è direttamente un parametro del filtro Butterworth standard in scipy.
            # Se la risonanza è cruciale, si dovrebbe usare un filtro biquad dinamico o un SVF.
            # Per ora, la risonanza sarà un parametro "non usato" per i filtri Butterworth,
            # o usiamo librosa.effects.band_biquad che ha un parametro Q.
            
            # Utilizziamo librosa.effects.band_biquad per un controllo migliore di Q
            # Richiede l'intero audio per volta, quindi dobbiamo ricalcolare a ogni chiamata
            # OPPURE lo applichiamo in batch più piccoli, ma è più complesso
            # L'approccio più semplice è calcolare i parametri per ogni frame e applicare un filtro statico.
            
            # Numero di campioni per blocco (corrisponde a un "frame" audio)
            samples_per_block = int(self.sample_rate / self.fps)
            
            # Calcola i coefficienti del filtro per il blocco corrente.
            # `librosa.effects.band_biquad` è una buona scelta per un filtro dinamico
            # ma richiede un array audio completo o blocchi.
            
            # Un approccio più efficiente per filtri dinamici con librosa è `librosa.effects.band_reject` etc.
            # che non sono dinamici.
            # Per filtri dinamici punto-per-punto, serve una propria implementazione o PyTorch/TensorFlow per DSP
            
            # Soluzione alternativa: applichiamo il filtro sul segnale *interpolato*
            # Questo è un compromesso tra un filtro davvero dinamico e la performance
            
            # Il metodo più "librosa-friendly" per filtri dinamici implica l'uso di filtri IIR
            # per segmenti o un design di filtro in tempo reale (complesso).
            # Se la variazione è lenta, possiamo ricalcolare i coefficienti ogni frame.
            
            # Invece di applicare sample per sample (che è inefficiente),
            # applichiamo i parametri del filtro per ogni "frame" logico del video.
            
        filtered_audio_blocks = []
        
        # Calcola i punti in cui ricalcolare il filtro
        frame_sample_indices = (np.arange(num_frames) * self.frame_to_sample_ratio).astype(int)
        frame_sample_indices = np.clip(frame_sample_indices, 0, len(audio_data) - 1)

        for i in range(num_frames):
            start_sample = frame_sample_indices[i]
            end_sample = frame_sample_indices[i+1] if i + 1 < num_frames else len(audio_data)
            
            if start_sample >= end_sample:
                continue # Salta blocchi vuoti

            block = audio_data[start_sample:end_sample]

            if len(block) == 0:
                continue

            # Applica _map_value solo al singolo valore interpolato, non all'array
            current_cutoff = _map_value(brightness_interp[start_sample], 0.0, 1.0, min_cutoff, max_cutoff)
            current_resonance = _map_value(detail_interp[start_sample], 0.0, 1.0, min_res, max_res)

            # Normalizza la frequenza di taglio per librosa
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = current_cutoff / nyquist

            # Assicurati che cutoff sia valido per il filtro
            if filter_type == "bandpass" or filter_type == "bandstop":
                # Per bandpass/bandstop, serve una banda. Per ora, simuliamo con un cutoff singolo
                # Se vogliamo una banda, dobbiamo definire anche una larghezza di banda.
                # Per semplicità, mappiamo un singolo cutoff.
                # Questo significa che i filtri bandpass/bandstop potrebbero non comportarsi come previsto
                # senza un secondo parametro.
                # Per ora, li useremo come se fossero dei lowpass/highpass centrati.
                # Se l'utente vuole un vero bandpass, dovrebbe specificare due frequenze.
                
                # Per evitare errori, facciamo un cutoff singolo per Butterworth
                # e useremo `librosa.effects.band_biquad` se è davvero un bandpass.
                # Ma librosa.effects.band_biquad opera sull'intero segnale.
                
                # Ritorno a scipy.signal.butter e lfilter per gestione blocco per blocco
                # Per Butterworth, Q non è un parametro diretto. Consideriamo solo il cutoff.
                if normal_cutoff >= 0.99: normal_cutoff = 0.99
                if normal_cutoff <= 0.01: normal_cutoff = 0.01

                b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
                filtered_block = lfilter(b, a, block)
            else: # lowpass, highpass
                if normal_cutoff >= 0.99: normal_cutoff = 0.99
                if normal_cutoff <= 0.01: normal_cutoff = 0.01
                
                b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
                filtered_block = lfilter(b, a, block)
            
            filtered_audio_blocks.append(filtered_block)
            
            if i % (num_frames // 100 + 1) == 0: # Aggiorna la progress bar
                progress = (i + 1) / num_frames
                progress_bar.progress(progress)
                status_text.text(f"Applicazione Filtro: {progress*100:.1f}% completata.")
        
        # Concatena tutti i blocchi filtrati
        # Assicurati che la lunghezza finale sia uguale a quella di input
        concatenated_audio = np.concatenate(filtered_audio_blocks)
        
        # Pad o taglia per adattarsi alla lunghezza originale
        if len(concatenated_audio) < len(audio_data):
            padded_audio = np.pad(concatenated_audio, (0, len(audio_data) - len(concatenated_audio)), 'constant')
            filtered_audio = padded_audio
        elif len(concatenated_audio) > len(audio_data):
            filtered_audio = concatenated_audio[:len(audio_data)]
        else:
            filtered_audio = concatenated_audio
            
        return filtered_audio

    def generate_fm_layer(self,
                          num_samples: int,
                          brightness_data: np.ndarray,
                          movement_data: np.ndarray, # Usato per modulare la frequenza del modulatore
                          min_carrier_freq: float,
                          max_carrier_freq: float,
                          min_modulator_freq: float,
                          max_modulator_freq: float,
                          min_mod_index: float,
                          max_mod_index: float,
                          progress_bar,
                          status_text) -> np.ndarray:
        """Genera un layer audio usando la sintesi FM."""
        fm_layer = np.zeros(num_samples, dtype=np.float32)
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False, dtype=np.float32)

        num_frames = len(brightness_data)
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, num_samples)

        carrier_freq_interp = np.interp(audio_indices, frames_indices, brightness_data)
        modulator_freq_interp = np.interp(audio_indices, frames_indices, movement_data)
        mod_index_interp = np.interp(audio_indices, frames_indices, detail_data) # Usiamo detail per l'indice

        # Applica _map_value a tutti gli array interpolati
        carrier_freq_interp = _map_value(carrier_freq_interp, 0.0, 1.0, min_carrier_freq, max_carrier_freq)
        modulator_freq_interp = _map_value(modulator_freq_interp, 0.0, 1.0, min_modulator_freq, max_modulator_freq)
        mod_index_interp = _map_value(mod_index_interp, 0.0, 1.0, min_mod_index, max_mod_index)
        
        # Evita divisioni per zero o valori estremi
        carrier_freq_interp = np.clip(carrier_freq_interp, 10.0, 20000.0)
        modulator_freq_interp = np.clip(modulator_freq_interp, 1.0, 1000.0)
        mod_index_interp = np.clip(mod_index_interp, 0.0, 100.0) # Limita l'indice di modulazione per prevenire overflow

        # Accumulatori di fase per evitare problemi di discontinuità
        phase_carrier = 0.0
        phase_modulator = 0.0
        
        samples_per_frame = int(self.sample_rate / self.fps)
        
        for i in range(num_samples):
            # Ottieni i valori interpolati per il sample corrente
            current_carrier_freq = carrier_freq_interp[i]
            current_modulator_freq = modulator_freq_interp[i]
            current_mod_index = mod_index_interp[i]

            # Calcola la fase del modulatore
            modulator_value = np.sin(phase_modulator)
            
            # Applica l'indice di modulazione alla frequenza del carrier
            # Il termine di modulazione è current_mod_index * modulator_value
            fm_freq = current_carrier_freq + current_mod_index * current_modulator_freq * modulator_value
            
            # Aggiorna la fase del carrier usando la frequenza modulata
            phase_carrier += 2 * np.pi * fm_freq / self.sample_rate
            
            # Aggiorna la fase del modulatore
            phase_modulator += 2 * np.pi * current_modulator_freq / self.sample_rate
            
            # Genera il sample FM
            fm_layer[i] = np.sin(phase_carrier)

            if i % samples_per_frame == 0:
                progress = i / num_samples
                progress_bar.progress(progress)
                status_text.text(f"Generazione FM: {progress*100:.1f}% completata.")
        
        progress_bar.progress(1.0)
        status_text.text("Generazione FM: 100% completata.")

        # Clip finale per prevenire overflow prima di sommare
        fm_layer = np.clip(fm_layer, -1.0, 1.0)
        
        return fm_layer

    def generate_granular_layer(self,
                                num_samples: int,
                                brightness_data: np.ndarray,
                                detail_data: np.ndarray,
                                movement_data: np.ndarray,
                                min_grain_freq: float,
                                max_grain_freq: float,
                                min_grain_density: float,
                                max_grain_density: float,
                                min_grain_duration: float,
                                max_grain_duration: float,
                                progress_bar,
                                status_text) -> np.ndarray:
        """Genera un layer audio usando la sintesi granulare."""
        granular_layer = np.zeros(num_samples, dtype=np.float32)
        
        num_frames = len(brightness_data)
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, num_samples)

        grain_freq_interp = np.interp(audio_indices, frames_indices, brightness_data)
        grain_density_interp = np.interp(audio_indices, frames_indices, movement_data)
        grain_duration_interp = np.interp(audio_indices, frames_indices, detail_data)

        # Applica _map_value a tutti gli array interpolati
        grain_freq_interp = _map_value(grain_freq_interp, 0.0, 1.0, min_grain_freq, max_grain_freq)
        grain_density_interp = _map_value(grain_density_interp, 0.0, 1.0, min_grain_density, max_grain_density)
        grain_duration_interp = _map_value(grain_duration_interp, 0.0, 1.0, min_grain_duration, max_grain_duration)

        # Assicurati che le durate siano positive
        grain_duration_interp = np.clip(grain_duration_interp, 0.001, None)

        current_sample = 0
        while current_sample < num_samples:
            
            # Scegli i parametri per il grano corrente in base all'interpolazione
            # Usa il valore del punto più vicino all'inizio del grano
            idx = int(current_sample / (num_samples / num_frames))
            idx = np.clip(idx, 0, num_frames - 1)

            freq = grain_freq_interp[current_sample]
            density_factor = grain_density_interp[current_sample]
            duration_s = grain_duration_interp[current_sample]

            # Calcola la durata in campioni
            grain_samples = int(duration_s * self.sample_rate)
            
            # Genera il grano (es. onda sinusoidale con inviluppo)
            grain_t = np.linspace(0, duration_s, grain_samples, endpoint=False, dtype=np.float32)
            grain_waveform = np.sin(2 * np.pi * freq * grain_t)

            # Applica un inviluppo (es. Hann window)
            window = np.hanning(grain_samples)
            grain = grain_waveform * window

            # Aggiungi il grano al layer principale
            end_grain_sample = current_sample + grain_samples
            
            # Gestisci il caso in cui il grano superi la fine dell'audio
            if end_grain_sample > num_samples:
                grain_to_add = grain[:num_samples - current_sample]
                granular_layer[current_sample:num_samples] += grain_to_add
            else:
                granular_layer[current_sample:end_grain_sample] += grain
            
            # Calcola il tempo al prossimo grano basato sulla densità
            # Una densità più alta significa che i grani si sovrappongono di più o sono più ravvicinati
            # Intervallo tra grani in secondi
            # Minore density_factor -> maggiore intervallo
            # Maggiore density_factor -> minore intervallo (più sovrapposizione)
            interval_s = duration_s * (1.0 - np.clip(density_factor, 0.0, 0.99)) # Evita 1.0 per non avere intervallo zero
            
            current_sample += int(interval_s * self.sample_rate) # Avanza al prossimo punto di partenza del grano

            if current_sample % int(num_samples / 100 + 1) == 0:
                progress = current_sample / num_samples
                progress_bar.progress(progress)
                status_text.text(f"Generazione Granulare: {progress*100:.1f}% completata.")
        
        progress_bar.progress(1.0)
        status_text.text("Generazione Granulare: 100% completata.")

        granular_layer = np.clip(granular_layer, -1.0, 1.0)
        return granular_layer

    def add_noise_layer(self,
                        audio_data: np.ndarray,
                        detail_data: np.ndarray,
                        min_noise_amp: float,
                        max_noise_amp: float,
                        progress_bar,
                        status_text) -> np.ndarray:
        """Aggiunge rumore bianco modulato dall'ampiezza in base al dettaglio."""
        
        num_frames = len(detail_data)
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, len(audio_data))

        noise_amp_interp = np.interp(audio_indices, frames_indices, detail_data)
        
        # Applicazione vettorializzata della mappatura
        # Old way: noise_amp_interp = _map_value(noise_amp_interp, 0.0, 1.0, min_noise_amp, max_noise_amp)
        # New way:
        clipped_noise_amp_interp = np.clip(noise_amp_interp, 0.0, 1.0)
        noise_amp_mapped = min_noise_amp + (clipped_noise_amp_interp - 0.0) * \
                           (max_noise_amp - min_noise_amp) / (1.0 - 0.0)
        noise_amp_interp = np.clip(noise_amp_mapped, 0.0, 1.0) # Clip finale tra 0 e 1 per ampiezza

        noise = np.random.uniform(-1.0, 1.0, len(audio_data)).astype(np.float32)
        modulated_noise = noise * noise_amp_interp

        # Applica una leggera media mobile sull'ampiezza del rumore per evitare click bruschi
        window_size = int(self.sample_rate / self.fps / 2) # Mezza finestra basata su mezzo frame
        if window_size % 2 == 0: window_size += 1 # Assicura che sia dispari per uniform_filter1d
        if window_size > 1:
            modulated_noise = uniform_filter1d(modulated_noise, size=window_size)
            
        combined_audio = audio_data + modulated_noise
        
        # Aggiorna progress bar (una sola volta per semplicità o per blocchi)
        progress_bar.progress(1.0)
        status_text.text("Aggiunta Rumore: 100% completata.")

        return np.clip(combined_audio, -1.0, 1.0)

    def apply_glitch_effect(self,
                            audio_data: np.ndarray,
                            variation_movement_data: np.ndarray,
                            glitch_threshold: float,
                            glitch_duration_frames: int,
                            glitch_intensity: float,
                            progress_bar,
                            status_text) -> np.ndarray:
        """Applica un effetto glitch all'audio basato sulla variazione del movimento."""
        glitched_audio = np.copy(audio_data)
        
        num_frames = len(variation_movement_data)
        samples_per_frame = int(self.sample_rate / self.fps)

        for i in range(num_frames):
            if variation_movement_data[i] > glitch_threshold:
                start_sample = i * samples_per_frame
                end_sample = min(start_sample + glitch_duration_frames * samples_per_frame, len(audio_data))
                
                if start_sample >= end_sample:
                    continue

                segment = glitched_audio[start_sample:end_sample]

                if len(segment) > 0:
                    # Inverti il segmento e applica intensità
                    glitched_audio[start_sample:end_sample] = segment[::-1] * (1.0 + glitch_intensity)
            
            if i % (num_frames // 100 + 1) == 0:
                progress_bar.progress((i + 1) / num_frames)
                status_text.text(f"Applicazione Glitch: {(i + 1) / num_frames * 100:.1f}% completata.")
        
        progress_bar.progress(1.0)
        status_text.text("Applicazione Glitch: 100% completata.")

        return np.clip(glitched_audio, -1.0, 1.0)

    def apply_pitch_time_stretch(self,
                                 audio_data: np.ndarray,
                                 brightness_data: np.ndarray,
                                 detail_data: np.ndarray,
                                 min_pitch_shift_semitones: float,
                                 max_pitch_shift_semitones: float,
                                 min_time_stretch_rate: float,
                                 max_time_stretch_rate: float,
                                 status_text) -> np.ndarray:
        """
        Applica pitch shifting e time stretching dinamici all'audio.
        Il pitch è controllato dalla luminosità, il time stretch dal dettaglio.
        """
        processed_audio = np.copy(audio_data)
        
        num_frames = len(brightness_data)
        samples_per_frame = int(self.sample_rate / self.fps)

        # Assicurati che audio_data sia mono per librosa.effects
        if processed_audio.ndim > 1:
            processed_audio = processed_audio.mean(axis=1) # Converti a mono

        # Interpola i dati di controllo per avere un valore per ogni "blocco" audio
        frame_indices = np.arange(num_frames)
        # Blocchi per l'elaborazione (es. 100ms per blocco)
        block_size_samples = int(self.sample_rate * 0.1) # 100 ms blocks
        
        stretched_segments = []
        original_samples_processed = 0

        for i in range(0, len(processed_audio), block_size_samples):
            current_block = processed_audio[i : i + block_size_samples]
            if len(current_block) == 0:
                continue

            # Calcola l'indice del frame video corrispondente all'inizio del blocco audio
            current_frame_idx = int((i / self.sample_rate) * self.fps)
            current_frame_idx = np.clip(current_frame_idx, 0, num_frames - 1)

            # Mappa i valori di controllo per il blocco corrente
            pitch_shift_semitones = _map_value(
                brightness_data[current_frame_idx],
                0.0, 1.0, min_pitch_shift_semitones, max_pitch_shift_semitones
            )
            time_stretch_rate = _map_value(
                detail_data[current_frame_idx],
                0.0, 1.0, min_time_stretch_rate, max_time_stretch_rate
            )
            
            # Applica lo stretch (librosa.effects.time_stretch può cambiare la lunghezza)
            # Utilizza hop_length calcolato dalla block_size
            hop_length = int(current_block.shape[0] / 4) # Generalmente una frazione del blocco
            if hop_length == 0: hop_length = 256 # Minimum hop length

            try:
                stretched_block = librosa.effects.time_stretch(current_block, rate=time_stretch_rate, hop_length=hop_length)
                
                # Applica il pitch shift
                # librosa.effects.pitch_shift usa un FFT, quindi il block deve avere una certa lunghezza
                # Se stretched_block è troppo corto, potrebbe dare errore.
                # Min_length per FFT è 2 * hop_length, spesso 2048 o 4096.
                if len(stretched_block) < 2048: # Scegli un valore ragionevole
                    # Padda il blocco se troppo corto per librosa
                    stretched_block = np.pad(stretched_block, (0, 2048 - len(stretched_block)), mode='constant')
                
                # Assicurati che non ci siano NaN/inf prima del pitch_shift
                if not np.all(np.isfinite(stretched_block)):
                    status_text.warning("⚠️ Blocco audio non finito prima del pitch_shift. Sostituzione con zeri.")
                    stretched_block[~np.isfinite(stretched_block)] = 0.0

                pitched_stretched_block = librosa.effects.pitch_shift(
                    stretched_block,
                    sr=self.sample_rate,
                    n_steps=pitch_shift_semitones
                )
                
                # Rimuovi il padding extra se applicato
                pitched_stretched_block = pitched_stretched_block[:len(stretched_block)] # Torna alla lunghezza del stretched_block originale
                
                stretched_segments.append(pitched_stretched_block)
                original_samples_processed += len(current_block)
            except Exception as e:
                status_text.error(f"❌ Errore durante l'applicazione di pitch/time stretch ad un blocco: {e}. Il blocco verrà saltato.")
                # Se c'è un errore, aggiungiamo un segmento di silenzio o il blocco originale
                stretched_segments.append(np.zeros_like(current_block, dtype=np.float32))
                original_samples_processed += len(current_block)


            progress = i / len(processed_audio)
            # progress_bar.progress(progress) # Aggiornamento costante potrebbe rallentare
            status_text.text(f"Applicazione Pitch/Time Stretch: {progress*100:.1f}% completata.")
        
        if stretched_segments:
            final_stretched_audio = np.concatenate(stretched_segments)
        else:
            final_stretched_audio = np.zeros_like(audio_data, dtype=np.float32)

        # Pad o taglia per adattarsi alla lunghezza originale
        if len(final_stretched_audio) < len(audio_data):
            final_stretched_audio = np.pad(final_stretched_audio, (0, len(audio_data) - len(final_stretched_audio)), 'constant')
        elif len(final_stretched_audio) > len(audio_data):
            final_stretched_audio = final_stretched_audio[:len(audio_data)]

        # Assicurati che l'output sia mono e pulito
        if final_stretched_audio.ndim > 1:
            final_stretched_audio = final_stretched_audio.mean(axis=1)

        # Controllo finale dei valori non finiti
        if not np.all(np.isfinite(final_stretched_audio)):
            status_text.warning("⚠️ L'audio finale dopo pitch/time stretch contiene NaN/inf. Sostituzione con zeri.")
            final_stretched_audio[~np.isfinite(final_stretched_audio)] = 0.0
            
        return final_stretched_audio


    def apply_delay_effect(self,
                           audio_data: np.ndarray,
                           movement_data: np.ndarray,
                           detail_data: np.ndarray,
                           max_delay_time_s: float,
                           max_delay_feedback: float,
                           progress_bar,
                           status_text) -> np.ndarray:
        """Applica un effetto delay dinamico."""
        
        delayed_audio = np.copy(audio_data)
        
        num_frames = len(movement_data)
        samples_per_frame = int(self.sample_rate / self.fps)

        audio_len_samples = len(audio_data)
        
        # Interpola i dati di controllo per l'intero array audio
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, audio_len_samples)

        delay_time_interp = np.interp(audio_indices, frames_indices, movement_data)
        delay_feedback_interp = np.interp(audio_indices, frames_indices, detail_data)

        # Mappa agli intervalli desiderati
        delay_time_interp = _map_value(delay_time_interp, 0.0, 1.0, 0.01, max_delay_time_s) # Min delay time
        delay_feedback_interp = _map_value(delay_feedback_interp, 0.0, 1.0, 0.0, max_delay_feedback)
        
        # Clipa feedback per evitare runaway (feedback > 1.0)
        delay_feedback_interp = np.clip(delay_feedback_interp, 0.0, 0.99)

        # Buffer per il delay (dimensione massima necessaria)
        max_delay_samples = int(max_delay_time_s * self.sample_rate) + 1
        delay_buffer = np.zeros(max_delay_samples, dtype=np.float32)
        buffer_ptr = 0 # Puntatore circolare

        for i in range(audio_len_samples):
            current_delay_time_s = delay_time_interp[i]
            current_delay_feedback = delay_feedback_interp[i]

            # Calcola il delay in campioni
            delay_samples = int(current_delay_time_s * self.sample_rate)
            delay_samples = np.clip(delay_samples, 0, max_delay_samples - 1) # Assicurati che sia nel buffer

            # Indice di lettura dal buffer
            read_idx = (buffer_ptr - delay_samples + max_delay_samples) % max_delay_samples
            
            # Leggi dal buffer
            delayed_signal = delay_buffer[read_idx]

            # Scrivi nel buffer (audio_in + feedback_del_segnale_ritardato)
            input_signal = audio_data[i]
            # Assicurati che input_signal sia finito
            if not np.isfinite(input_signal):
                input_signal = 0.0
                
            signal_to_buffer = input_signal + delayed_signal * current_delay_feedback
            delay_buffer[buffer_ptr] = signal_to_buffer
            
            # Aggiungi il segnale ritardato all'output (dry/wet mix implicito qui, potresti aggiungere un dry/wet knob)
            delayed_audio[i] = input_signal + delayed_signal # Mix secco/umido semplice

            # Aggiorna il puntatore del buffer
            buffer_ptr = (buffer_ptr + 1) % max_delay_samples
            
            if i % (audio_len_samples // 100 + 1) == 0:
                progress_bar.progress(i / audio_len_samples)
                status_text.text(f"Applicazione Delay: {i / audio_len_samples * 100:.1f}% completata.")
        
        progress_bar.progress(1.0)
        status_text.text("Applicazione Delay: 100% completata.")

        return np.clip(delayed_audio, -1.0, 1.0)

    def apply_reverb_effect(self,
                            audio_data: np.ndarray,
                            detail_data: np.ndarray, # Per modulare il decadimento
                            brightness_data: np.ndarray, # Per modulare il wet level
                            max_reverb_decay_s: float,
                            max_reverb_wet_level: float,
                            progress_bar,
                            status_text) -> np.ndarray:
        """Applica un effetto riverbero dinamico (simulato con delay multipli)."""
        # Per un vero riverbero, servirebbe una convoluzione con una Impulse Response (IR).
        # Per semplicità e performance, simuliamo un riverbero con un insieme di delay con feedback.
        # Questa non è una vera riverberazione ma un effetto "spaziale".

        reverbed_audio = np.zeros_like(audio_data, dtype=np.float32)

        num_frames = len(detail_data)
        
        # Interpola i dati di controllo per l'intero array audio
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, len(audio_data))

        decay_interp = np.interp(audio_indices, frames_indices, detail_data)
        wet_interp = np.interp(audio_indices, frames_indices, brightness_data)

        decay_interp = _map_value(decay_interp, 0.0, 1.0, 0.1, max_reverb_decay_s)
        wet_interp = _map_value(wet_interp, 0.0, 1.0, 0.0, max_reverb_wet_level)
        
        # Clipa i valori
        decay_interp = np.clip(decay_interp, 0.1, 5.0) # Decay da 0.1s a 5s
        wet_interp = np.clip(wet_interp, 0.0, 1.0)

        # Simulazione riverbero con più linee di delay (es. 4-5 linee)
        num_delay_lines = 4
        # Tempi di delay primi (non multipli)
        delay_times = np.array([0.029, 0.037, 0.043, 0.051]) # Tempi in secondi
        
        delay_buffers = [np.zeros(int(t * self.sample_rate) + 1, dtype=np.float32) for t in delay_times]
        buffer_pointers = [0] * num_delay_lines

        for i in range(len(audio_data)):
            current_decay = decay_interp[i]
            current_wet = wet_interp[i]

            # Calcola il feedback per ogni linea di delay in base al decadimento globale
            # Un decadimento più lungo implica un feedback più alto
            # Feedback come e^(-1/tau) dove tau è il tempo di decadimento
            current_feedback_factor = 1.0 - (1.0 / (current_decay * self.sample_rate + 1e-6)) # Approssimazione
            current_feedback_factor = np.clip(current_feedback_factor, 0.0, 0.99) # Impedisce runaway

            wet_signal = 0.0
            input_signal = audio_data[i]
            
            # Assicurati che input_signal sia finito
            if not np.isfinite(input_signal):
                input_signal = 0.0

            for j in range(num_delay_lines):
                delay_buffer = delay_buffers[j]
                buffer_ptr = buffer_pointers[j]
                
                delay_samples = int(delay_times[j] * self.sample_rate)
                delay_samples = np.clip(delay_samples, 0, len(delay_buffer) - 1)

                read_idx = (buffer_ptr - delay_samples + len(delay_buffer)) % len(delay_buffer)
                
                delayed_val = delay_buffer[read_idx]
                
                # Aggiungi il segnale di input al buffer con feedback
                signal_to_buffer = input_signal + delayed_val * current_feedback_factor
                delay_buffer[buffer_ptr] = signal_to_buffer

                wet_signal += delayed_val # Aggiungi al segnale umido

                buffer_pointers[j] = (buffer_ptr + 1) % len(delay_buffer)
            
            # Mix dry/wet
            reverbed_audio[i] = audio_data[i] * (1.0 - current_wet) + wet_signal * current_wet

            if i % (len(audio_data) // 100 + 1) == 0:
                progress_bar.progress(i / len(audio_data))
                status_text.text(f"Applicazione Riverbero: {i / len(audio_data) * 100:.1f}% completata.")
        
        progress_bar.progress(1.0)
        status_text.text("Applicazione Riverbero: 100% completata.")

        return np.clip(reverbed_audio, -1.0, 1.0)

    def apply_modulation_effect(self,
                                audio_data: np.ndarray,
                                variation_movement_data: np.ndarray,
                                detail_data: np.ndarray,
                                progress_bar,
                                status_text,
                                effect_type: str = "tremolo",
                                intensity: float = 0.5, # Base intensity
                                rate: float = 0.1) -> np.ndarray: # Base rate (Hz)
        """Applica un effetto di modulazione dinamico (es. Tremolo)."""
        # Per ora implementiamo solo un Tremolo semplice.
        # Chorus/Flanger/Phaser richiederebbero implementazioni più complesse (delay modulati)
        modulated_audio = np.copy(audio_data)
        
        num_frames = len(variation_movement_data)
        samples_per_frame = int(self.sample_rate / self.fps)

        audio_len_samples = len(audio_data)
        
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, audio_len_samples)

        # Moduliamo intensità e rate in base ai dati video
        # Esempio: Variazione Movimento per il rate, Dettaglio per l'intensità
        modulated_intensity_interp = np.interp(audio_indices, frames_indices, detail_data)
        modulated_rate_interp = np.interp(audio_indices, frames_indices, variation_movement_data)

        # Mappiamo ai valori utente + modulazione
        # Minima intensità / rate per evitare 0
        modulated_intensity_interp = _map_value(modulated_intensity_interp, 0.0, 1.0, 0.1, intensity)
        modulated_rate_interp = _map_value(modulated_rate_interp, 0.0, 1.0, 0.05, rate * 10) # Fino a 10 volte il rate base

        if effect_type == "tremolo":
            # Tremolo: modulazione dell'ampiezza
            # LFO (Low Frequency Oscillator)
            lfo_phase = 0.0
            for i in range(audio_len_samples):
                current_rate = modulated_rate_interp[i]
                current_intensity = modulated_intensity_interp[i]

                # Genera l'onda LFO (es. sinusoide)
                lfo_value = np.sin(lfo_phase) * current_intensity
                
                # Modula l'ampiezza
                # La modulazione dovrebbe variare tra 1-intensità e 1+intensità
                amplitude_mod = 1.0 + lfo_value # Va da 1-int a 1+int
                
                modulated_audio[i] = audio_data[i] * np.clip(amplitude_mod, 0.0, 2.0) # Clip per evitare valori negativi/eccessivi

                # Aggiorna la fase dell'LFO
                lfo_phase += 2 * np.pi * current_rate / self.sample_rate
                
                if i % (audio_len_samples // 100 + 1) == 0:
                    progress_bar.progress(i / audio_len_samples)
                    status_text.text(f"Applicazione Tremolo: {i / audio_len_samples * 100:.1f}% completata.")
        # Puoi aggiungere qui if/elif per chorus, flanger, phaser
        # Questi richiederebbero un delay buffer e la modulazione del tempo di delay.
        # Es: Chorus: breve delay modulato, mixato con dry
        # Flanger: delay ancora più breve, con feedback e modulazione profonda
        # Phaser: filtro all-pass con fase modulata (più complesso)
        else:
            status_text.warning(f"Effetto di modulazione '{effect_type}' non implementato.")

        progress_bar.progress(1.0)
        status_text.text(f"Applicazione Modulazione: 100% completata.")

        return np.clip(modulated_audio, -1.0, 1.0)


    def apply_stereo_panning(self, audio_data: np.ndarray, horizontal_center_data: np.ndarray) -> np.ndarray:
        """Applica un panning stereo dinamico all'audio."""
        # Se l'audio è già stereo, convertilo a mono prima di riapplicare il panning
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        num_samples = len(audio_data)
        num_frames = len(horizontal_center_data)

        # Interpola i dati di panning per ogni sample
        frames_indices = np.arange(num_frames)
        audio_indices = np.linspace(0, num_frames - 1, num_samples)
        panning_interp = np.interp(audio_indices, frames_indices, horizontal_center_data)

        # Variazione dei coefficienti di gain per i canali sinistro e destro
        # 0.0 = sinistra, 0.5 = centro, 1.0 = destra
        # Per la legge di potenza (-3dB pan law), gain = cos(angle), sin(angle)
        # angle = panning_interp * pi / 2 (0 a pi/2)
        angle = panning_interp * (np.pi / 2)
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        # Crea un array stereo vuoto
        stereo_audio = np.zeros((num_samples, 2), dtype=np.float32)

        stereo_audio[:, 0] = audio_data * left_gain
        stereo_audio[:, 1] = audio_data * right_gain

        return stereo_audio

    def normalize_audio(self, audio_data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizza l'audio al picco desiderato."""
        if audio_data.size == 0:
            return audio_data

        abs_max = np.max(np.abs(audio_data))
        if abs_max == 0:
            return audio_data # Evita divisione per zero se l'audio è silenzioso
        
        normalized_audio = audio_data * (target_peak / abs_max)
        return np.clip(normalized_audio, -1.0, 1.0) # Clipa per sicurezza

# --- Funzione Main di Streamlit ---
def main():
    st.set_page_config(page_title="🎵 VideoSound Gen - Sperimentale", layout="centered")
    st.title("🎬 VideoSound Gen - Sperimentale")
    st.markdown("###### by Loop507")
    st.markdown("### Genera musica sperimentale da un video o da parametri manuali!")

    # --- NUOVA SEZIONE: Scelta della Sorgente di Input ---
    input_mode = st.radio(
        "Seleziona la sorgente per la generazione del suono:",
        ("Carica un Video", "Genera Suono Senza Video (Parametri Manuali)"),
        index=0, # "Carica un Video" è l'opzione predefinita
        help="Scegli se analizzare un video per generare il suono, o se creare un suono basato su parametri che imposterai manualmente."
    )

    uploaded_file = None
    video_input_path = None

    # Inizializzazione delle variabili che conterranno i dati di modulazione
    brightness_data = None
    detail_data = None
    movement_data = None
    variation_movement_data = None
    horizontal_center_data = None

    # Inizializzazione delle informazioni sul "video" (virtuale o reale)
    width, height, fps, video_duration = 0, 0, 0.0, 0.0

    # --- LOGICA IN BASE ALLA SCELTA DELL'UTENTE ---
    if input_mode == "Carica un Video":
        uploaded_file = st.file_uploader("🎞️ Carica un file video (.mp4, .mov, ecc.)", type=["mp4", "mov", "avi", "mkv"])

        if uploaded_file is not None:
            if not validate_video_file(uploaded_file):
                return

            base_name_upload = os.path.splitext(uploaded_file.name)[0]
            unique_id = str(np.random.randint(10000, 99999))
            video_input_path = os.path.join("temp", f"{base_name_upload}_{unique_id}.mp4")
            os.makedirs("temp", exist_ok=True)
            with open(video_input_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("🎥 Video caricato correttamente!")

            analysis_progress_bar = st.progress(0)
            analysis_status_text = st.empty()

            with st.spinner("📊 Analisi frame video (luminosità, dettaglio, movimento, variazione movimento, centro orizzontale) in corso..."):
                brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, video_duration = analyze_video_frames(video_input_path, analysis_progress_bar, analysis_status_text)

            if brightness_data is None:
                st.error("❌ Errore durante l'analisi del video. Riprova con un altro file.")
                return # Si è verificato un errore nell'analisi del video

            st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")
        else:
            st.info("⬆️ Carica un video per iniziare o cambia modalità di generazione.")
            return # Ferma l'esecuzione se non c'è input video

    elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
        st.info("ℹ️ Generazione audio basata su parametri manuali. Regola i controlli qui sotto per modellare il tuo suono.")

        st.subheader("⚙️ Configurazione Parametri 'Visivi' Virtuali")
        # Permetti all'utente di definire la durata dell'audio
        virtual_duration = st.slider("Durata Audio Generato (secondi)", 5, MAX_DURATION, 30, help="Definisci per quanto tempo l'audio verrà generato.")
        
        # Definisci i "frames per secondo" virtuali (possono essere gli stessi di AUDIO_FPS)
        virtual_fps = AUDIO_FPS
        num_virtual_frames = int(virtual_duration * virtual_fps)

        # Controlli per i "dati visivi" virtuali
        st.markdown("---")
        st.markdown("**Andamento Luminosità (controlla cutoff filtro, carrier FM, pitch, ecc.)**")
        start_brightness = st.slider("Luminosità Iniziale (0.0=scuro, 1.0=chiaro)", 0.0, 1.0, 0.2, 0.05, key="sb")
        end_brightness = st.slider("Luminosità Finale (0.0=scuro, 1.0=chiaro)", 0.0, 1.0, 0.8, 0.05, key="eb")
        brightness_data = np.linspace(start_brightness, end_brightness, num_virtual_frames)
        # Assicurati che i dati siano finiti
        if not np.all(np.isfinite(brightness_data)):
            st.error("❌ Errore: I dati di luminosità virtuali contengono valori non finiti. Impostazione a zero.")
            brightness_data[~np.isfinite(brightness_data)] = 0.0

        st.markdown("---")
        st.markdown("**Andamento Dettaglio/Contrasto (controlla risonanza, modulator FM, noise, ecc.)**")
        start_detail = st.slider("Dettaglio Iniziale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.8, 0.05, key="sd")
        end_detail = st.slider("Dettaglio Finale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.2, 0.05, key="ed")
        detail_data = np.linspace(start_detail, end_detail, num_virtual_frames)
        # Assicurati che i dati siano finiti
        if not np.all(np.isfinite(detail_data)):
            st.error("❌ Errore: I dati di dettaglio virtuali contengono valori non finiti. Impostazione a zero.")
            detail_data[~np.isfinite(detail_data)] = 0.0

        st.markdown("---")
        st.markdown("**Andamento Movimento (controlla glitch, delay, densità granulare, ecc.)**")
        base_movement = np.linspace(0.1, 0.5, num_virtual_frames) # Movimento di base crescente
        random_movement_factor = st.slider("Fattore di Movimento Casuale", 0.0, 1.0, 0.2, 0.05, help="Aggiunge casualità all'andamento del movimento.")
        movement_data = base_movement + (np.random.rand(num_virtual_frames) - 0.5) * random_movement_factor
        movement_data = np.clip(movement_data, 0.0, 1.0) # Assicura che i valori rimangano tra 0 e 1
        variation_movement_data = np.abs(np.diff(movement_data, prepend=movement_data[0]))
        # Assicurati che i dati siano finiti
        if not np.all(np.isfinite(movement_data)):
            st.error("❌ Errore: I dati di movimento virtuali contengono valori non finiti. Impostazione a zero.")
            movement_data[~np.isfinite(movement_data)] = 0.0
        if not np.all(np.isfinite(variation_movement_data)):
            st.error("❌ Errore: I dati di variazione movimento virtuali contengono valori non finiti. Impostazione a zero.")
            variation_movement_data[~np.isfinite(variation_movement_data)] = 0.0

        st.markdown("---")
        st.markdown("**Andamento Panning Orizzontale (controlla il bilanciamento stereo)**")
        start_pan = st.slider("Panning Iniziale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="sp")
        end_pan = st.slider("Panning Finale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="ep")
        horizontal_center_data = np.linspace(start_pan, end_pan, num_virtual_frames)
        # Assicurati che i dati siano finiti
        if not np.all(np.isfinite(horizontal_center_data)):
            st.error("❌ Errore: I dati di panning virtuali contengono valori non finiti. Impostazione a centro.")
            horizontal_center_data[~np.isfinite(horizontal_center_data)] = 0.5

        # Imposta i valori per il resto del codice, come se venissero da un video
        video_duration = float(virtual_duration)
        fps = float(virtual_fps)
        width, height = 1280, 720 # Valori placeholder, non usati per la generazione audio

        st.success(f"✅ Dati 'visivi' virtuali generati per {virtual_duration} secondi di audio.")

    # Se a questo punto i dati non sono stati inizializzati (l'utente non ha ancora caricato un video
    # o scelto di generare con parametri manuali), fermiamo l'esecuzione.
    if brightness_data is None:
        return
        
    st.markdown("---")
    st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

    # --- Inizializzazione dei parametri audio di default (prima della UI) ---
    # Questo assicura che esistano sempre, indipendentemente dai checkbox abilitati
    min_cutoff_user, max_cutoff_user = 0, 0
    min_resonance_user, max_resonance_user = 0, 0
    waveform_type_user = "sawtooth"
    filter_type_user = "lowpass"

    min_carrier_freq_user, max_carrier_freq_user = 0, 0
    min_modulator_freq_user, max_modulator_freq_user = 0, 0
    min_mod_index_user, max_mod_index_user = 0, 0

    min_noise_amp_user, max_noise_amp_user = 0, 0
    glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user = 0, 0, 0
    min_pitch_shift_semitones = 0
    max_pitch_shift_semitones = 0
    min_time_stretch_rate, max_time_stretch_rate = 0, 0
    min_grain_freq, max_grain_freq = 0, 0
    min_grain_density, max_grain_density = 0, 0
    min_grain_duration, max_grain_duration = 0, 0
    max_delay_time_user, max_delay_feedback_user = 0, 0
    max_reverb_decay_user, max_reverb_wet_user = 0, 0
    modulation_effect_type = "none"
    modulation_intensity = 0.5
    modulation_rate = 0.1

    min_cutoff_adv = 20
    max_cutoff_adv = 20000
    min_resonance_adv = 0.1
    max_resonance_adv = 30.0

    # --- Sezione Sintesi Sottrattiva (con checkbox di abilitazione) ---
    st.sidebar.header("Generazione Suono Base")
    enable_subtractive_synthesis = st.sidebar.checkbox("🔊 **Abilita Sintesi Sottrattiva (Suono Base)**", value=True)
    with st.sidebar.expander("Sintesi Sottrattiva (Forma d'Onda & Filtro)", expanded=True):
        st.markdown("#### Oscillatore")
        waveform_type_user = st.selectbox(
            "Forma d'Onda Oscillatore:",
            ("sawtooth", "square", "sine", "triangle"),
            key="waveform_type",
            disabled=not enable_subtractive_synthesis
        )
        st.markdown("#### Filtro Base (modulato dalla Luminosità/Dettaglio)")
        st.markdown("**Controlli:**")
        st.markdown("- **Frequenza di Taglio:** controllata dalla **Luminosità** del video/input.")
        st.markdown("- **Risonanza:** controllata dal **Dettaglio/Contrasto** del video/input.")
        min_cutoff_user = st.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100, key="sub_min_cutoff", disabled=not enable_subtractive_synthesis)
        max_cutoff_user = st.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000, key="sub_max_cutoff", disabled=not enable_subtractive_synthesis)
        min_resonance_user = st.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5, key="sub_min_res", disabled=not enable_subtractive_synthesis)
        max_resonance_user = st.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0, key="sub_max_res", disabled=not enable_subtractive_synthesis)

    # --- Sezione Sintesi FM ---
    st.sidebar.header("Sintesi FM")
    enable_fm_synthesis = st.sidebar.checkbox("📡 **Abilita Sintesi FM**", value=False)
    with st.sidebar.expander("Sintesi FM (Carrier, Modulator, Index)", expanded=False):
        st.markdown("**Controlli:**")
        st.markdown("- **Frequenza Carrier:** controllata dalla **Luminosità**.")
        st.markdown("- **Frequenza Modulatore:** controllata dal **Movimento**.")
        st.markdown("- **Indice di Modulazione:** controllato dal **Dettaglio/Contrasto**.")
        min_carrier_freq_user = st.slider("Min Frequenza Carrier (Hz)", 50, 1000, 100, key="fm_min_carr", disabled=not enable_fm_synthesis)
        max_carrier_freq_user = st.slider("Max Frequenza Carrier (Hz)", 500, 5000, 1500, key="fm_max_carr", disabled=not enable_fm_synthesis)
        min_modulator_freq_user = st.slider("Min Frequenza Modulatore (Hz)", 1, 100, 5, key="fm_min_mod", disabled=not enable_fm_synthesis)
        max_modulator_freq_user = st.slider("Max Frequenza Modulatore (Hz)", 10, 500, 100, key="fm_max_mod", disabled=not enable_fm_synthesis)
        min_mod_index_user = st.slider("Min Indice Modulazione", 0.0, 5.0, 0.1, key="fm_min_idx", disabled=not enable_fm_synthesis)
        max_mod_index_user = st.slider("Max Indice Modulazione", 1.0, 30.0, 10.0, key="fm_max_idx", disabled=not enable_fm_synthesis)

    # --- Sezione Sintesi Granulare ---
    st.sidebar.header("Sintesi Granulare")
    enable_granular_synthesis = st.sidebar.checkbox("🍚 **Abilita Sintesi Granulare**", value=False)
    with st.sidebar.expander("Sintesi Granulare (Frequenza, Densità, Durata Grano)", expanded=False):
        st.markdown("**Controlli:**")
        st.markdown("- **Frequenza Grani:** controllata dalla **Luminosità**.")
        st.markdown("- **Densità Grani:** controllata dal **Movimento**.")
        st.markdown("- **Durata Grani:** controllata dal **Dettaglio/Contrasto**.")
        min_grain_freq = st.slider("Min Frequenza Grano (Hz)", 10, 500, 50, key="gran_min_freq", disabled=not enable_granular_synthesis)
        max_grain_freq = st.slider("Max Frequenza Grano (Hz)", 100, 2000, 500, key="gran_max_freq", disabled=not enable_granular_synthesis)
        min_grain_density = st.slider("Min Densità Grani", 0.1, 5.0, 1.0, key="gran_min_dens", disabled=not enable_granular_synthesis)
        max_grain_density = st.slider("Max Densità Grani", 1.0, 20.0, 5.0, key="gran_max_dens", disabled=not enable_granular_synthesis)
        min_grain_duration = st.slider("Min Durata Grano (s)", 0.001, 0.1, 0.01, key="gran_min_dur", disabled=not enable_granular_synthesis)
        max_grain_duration = st.slider("Max Durata Grano (s)", 0.01, 0.5, 0.1, key="gran_max_dur", disabled=not enable_granular_synthesis)

    # --- Sezione Effetto Rumore ---
    st.sidebar.header("Effetto Rumore")
    enable_noise_effect = st.sidebar.checkbox("🌧️ **Abilita Rumore**", value=False)
    with st.sidebar.expander("Effetto Rumore (Ampiezza)", expanded=False):
        st.markdown("**Controllo:**")
        st.markdown("- **Ampiezza Rumore:** controllata dal **Dettaglio/Contrasto**.")
        min_noise_amp_user = st.slider("Min Ampiezza Rumore", 0.0, 0.5, 0.05, key="noise_min_amp", disabled=not enable_noise_effect)
        max_noise_amp_user = st.slider("Max Ampiezza Rumore", 0.1, 1.0, 0.3, key="noise_max_amp", disabled=not enable_noise_effect)

    # --- Sezione Effetto Glitch ---
    st.sidebar.header("Effetto Glitch")
    enable_glitch_effect = st.sidebar.checkbox("👾 **Abilita Glitch**", value=False)
    with st.sidebar.expander("Effetto Glitch (Soglia, Durata, Intensità)", expanded=False):
        st.markdown("**Controllo:**")
        st.markdown("- **Trigger Glitch:** basato sulla **Variazione del Movimento**.")
        glitch_threshold_user = st.slider("Soglia Variazione Movimento (per attivare glitch)", 0.0, 0.5, 0.1, key="glitch_thresh", disabled=not enable_glitch_effect)
        glitch_duration_frames_user = st.slider("Durata Glitch (frames)", 1, 10, 3, key="glitch_dur", disabled=not enable_glitch_effect)
        glitch_intensity_user = st.slider("Intensità Glitch", 0.0, 1.0, 0.5, key="glitch_int", disabled=not enable_glitch_effect)

    # --- Sezione Pitch Shifting e Time Stretching ---
    st.sidebar.header("Pitch & Time")
    enable_pitch_time_stretch = st.sidebar.checkbox("⏱️ **Abilita Pitch Shifting / Time Stretching**", value=False)
    with st.sidebar.expander("Pitch Shifting / Time Stretching", expanded=False):
        st.markdown("**Controlli:**")
        st.markdown("- **Pitch Shift:** controllato dalla **Luminosità**.")
        st.markdown("- **Time Stretch:** controllato dal **Dettaglio/Contrasto**.")
        min_pitch_shift_semitones = st.slider("Min Pitch Shift (semitoni)", -24, 0, -12, key="min_pitch", disabled=not enable_pitch_time_stretch)
        max_pitch_shift_semitones = st.slider("Max Pitch Shift (semitoni)", 0, 24, 12, key="max_pitch", disabled=not enable_pitch_time_stretch)
        min_time_stretch_rate = st.slider("Min Time Stretch Rate (più lento)", 0.1, 1.0, 0.5, key="min_time_stretch", disabled=not enable_pitch_time_stretch)
        max_time_stretch_rate = st.slider("Max Time Stretch Rate (più veloce)", 1.0, 5.0, 2.0, key="max_time_stretch", disabled=not enable_pitch_time_stretch)

    # --- Sezione Effetti Dinamici Avanzati (Delay, Reverb, Modulazione) ---
    st.sidebar.header("Effetti Dinamici Avanzati")
    enable_dynamic_effects = st.sidebar.checkbox("🌌 **Abilita Effetti Dinamici Avanzati**", value=False)
    with st.sidebar.expander("Filtro Avanzato, Modulazione, Delay, Riverbero", expanded=False):
        st.markdown("#### Filtro Avanzato (Globale - su tutto l'audio combinato)")
        filter_type_user = st.selectbox(
            "Tipo di Filtro:",
            ("lowpass", "highpass", "bandpass", "bandstop"),
            key="filter_type_adv",
            disabled=not enable_dynamic_effects
        )
        st.markdown("**Controlli:**")
        st.markdown("- **Frequenza di Taglio:** controllata dalla **Luminosità**.")
        st.markdown("- **Risonanza (Q):** controllata dal **Dettaglio/Contrasto**.")
        min_cutoff_adv = st.slider("Min Cutoff (Hz)", 20, 10000, 100, key="adv_min_cutoff", disabled=not enable_dynamic_effects)
        max_cutoff_adv = st.slider("Max Cutoff (Hz)", 1000, 20000, 15000, key="adv_max_cutoff", disabled=not enable_dynamic_effects)
        min_resonance_adv = st.slider("Min Risonanza Q", 0.1, 5.0, 0.7, key="adv_min_res", disabled=not enable_dynamic_effects)
        max_resonance_adv = st.slider("Max Risonanza Q", 1.0, 30.0, 15.0, key="adv_max_res", disabled=not enable_dynamic_effects)

        st.markdown("#### Effetto di Modulazione (Chorus/Flanger/Tremolo)")
        modulation_effect_type = st.selectbox(
            "Tipo di Modulazione:",
            ("none", "tremolo"), # Chorus/Flanger/Phaser sono placeholder nel tuo codice precedente
            key="mod_type",
            disabled=not enable_dynamic_effects
        )
        modulation_intensity = st.slider("Intensità Modulazione", 0.0, 1.0, 0.5, key="mod_int", disabled=not enable_dynamic_effects or modulation_effect_type == "none")
        modulation_rate = st.slider("Rate Modulazione (Hz)", 0.01, 10.0, 0.1, key="mod_rate", disabled=not enable_dynamic_effects or modulation_effect_type == "none")
        st.markdown("  *Note:* Intensità e rate di modulazione sono mappati alla **Variazione Movimento** e **Dettaglio**.")


        st.markdown("#### Delay (Eco)")
        max_delay_time_user = st.slider("Max Tempo Delay (s)", 0.0, 2.0, 0.5, key="delay_time", disabled=not enable_dynamic_effects)
        max_delay_feedback_user = st.slider("Max Feedback Delay", 0.0, 0.9, 0.4, key="delay_feedback", disabled=not enable_dynamic_effects)
        st.markdown("  *Note:* Tempo e feedback sono modulati da **Movimento** e **Dettaglio**.")

        st.markdown("#### Riverbero")
        max_reverb_decay_user = st.slider("Max Tempo Decadimento Riverbero (s)", 0.0, 5.0, 2.0, key="reverb_decay", disabled=not enable_dynamic_effects)
        max_reverb_wet_user = st.slider("Max Livello Wet Riverbero", 0.0, 1.0, 0.3, key="reverb_wet", disabled=not enable_dynamic_effects)
        st.markdown("  *Note:* Decadimento e livello wet sono modulati da **Dettaglio** e **Luminosità**.")

    st.markdown("---")
    st.subheader("⬇️ Opzioni di Download")

    output_resolution_choice = "Originale" # Default, sarà sovrascritto
    download_option_choices = ["Solo Audio"] # Default, se non c'è video

    if input_mode == "Carica un Video":
        output_resolution_choice = st.selectbox(
            "Seleziona la risoluzione di output del video:",
            list(FORMAT_RESOLUTIONS.keys())
        )
        download_option_choices = ["Video con Audio", "Solo Audio"]
    elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
        st.warning("⚠️ Stai generando solo audio. Le opzioni video non sono disponibili in questa modalità.")
        output_resolution_choice = "Originale" # Non usato, ma per coerenza
        download_option_choices = ["Solo Audio"]

    download_option = st.radio(
        "Cosa vuoi scaricare?",
        download_option_choices
    )

    if not check_ffmpeg():
        st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. L'unione o la ricodifica del video potrebbe non funzionare. Assicurati che FFmpeg sia installato e nel PATH.")

    if st.button("🎵 Genera e Prepara Download"):
        
        base_name_output = "generated_sound"
        if input_mode == "Carica un Video" and uploaded_file is not None:
            base_name_output = os.path.splitext(uploaded_file.name)[0]
        elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
            base_name_output = "manual_params_sound" # Nome specifico per output manuale

        unique_id_audio = str(np.random.randint(10000, 99999))
        audio_output_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_generated_audio.wav")
        final_video_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_final_videosound.mp4")

        os.makedirs("temp", exist_ok=True)

        audio_gen = AudioGenerator(AUDIO_SAMPLE_RATE, int(fps))

        total_samples = int(video_duration * AUDIO_SAMPLE_RATE)

        audio_progress_bar = st.progress(0)
        audio_status_text = st.empty()

        # --- Generazione dell'audio base (Sintesi Sottrattiva) ---
        if enable_subtractive_synthesis:
            st.info("🎵 Generazione dell'onda base (Sintesi Sottrattiva)...")
            subtractive_layer = audio_gen.generate_subtractive_waveform(total_samples, waveform_type_user)
            combined_audio_layers = subtractive_layer
            # Controllo: Sintesi Sottrattiva
            if not np.all(np.isfinite(combined_audio_layers)):
                st.error("❌ Errore: la sintesi sottrattiva ha prodotto valori non finiti. Sostituzione con zeri.")
                combined_audio_layers[~np.isfinite(combined_audio_layers)] = 0.0 # Sostituisci non-finiti con zero
            st.success("✅ Strato Sintesi Sottrattiva generato!")
        else:
            combined_audio_layers = np.zeros(total_samples, dtype=np.float32) # Assicura che sia inizializzato
            st.info("🎵 Sintesi Sottrattiva disabilitata.")


        # --- Aggiungi lo strato FM se abilitato ---
        if enable_fm_synthesis:
            st.info("🎵 Generazione dello strato FM...")
            fm_layer = audio_gen.generate_fm_layer(
                total_samples,
                brightness_data,
                movement_data, # Per l'FM, il movimento influenza la frequenza del modulatore
                min_carrier_freq_user, max_carrier_freq_user,
                min_modulator_freq_user, max_modulator_freq_user,
                min_mod_index_user, max_mod_index_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            # Clip per prevenire overflow prima di aggiungere
            fm_layer = np.clip(fm_layer, -1.0, 1.0)
            combined_audio_layers += fm_layer * 0.5 # Aggiunge con un certo mix
            # Controllo: Aggiunta FM
            if not np.all(np.isfinite(combined_audio_layers)):
                st.error("❌ Errore: l'aggiunta del layer FM ha introdotto valori non finiti. Sostituzione con zeri.")
                combined_audio_layers[~np.isfinite(combined_audio_layers)] = 0.0
            st.success("✅ Strato FM generato e combinato!")

        # --- Aggiungi lo strato Granulare se abilitato ---
        if enable_granular_synthesis:
            st.info("🎵 Generazione dello strato Granulare...")
            granular_layer = audio_gen.generate_granular_layer(
                total_samples,
                brightness_data,
                detail_data,
                movement_data,
                min_grain_freq, max_grain_freq,
                min_grain_density, max_grain_density,
                min_grain_duration, max_grain_duration,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            # Clip per prevenire overflow prima di aggiungere
            granular_layer = np.clip(granular_layer, -1.0, 1.0)
            combined_audio_layers += granular_layer * 0.5 # Aggiunge con un certo mix
            # Controllo: Aggiunta Granulare
            if not np.all(np.isfinite(combined_audio_layers)):
                st.error("❌ Errore: l'aggiunta del layer Granulare ha introdotto valori non finiti. Sostituzione con zeri.")
                combined_audio_layers[~np.isfinite(combined_audio_layers)] = 0.0
            st.success("✅ Strato Granulare generato e combinato!")

        # --- Aggiungi il layer di Rumore se abilitato ---
        if enable_noise_effect:
            st.info("🎵 Aggiunta del layer Rumore...")
            combined_audio_layers = audio_gen.add_noise_layer(
                combined_audio_layers,
                detail_data,
                min_noise_amp_user,
                max_noise_amp_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            # Controllo: Aggiunta Rumore
            if not np.all(np.isfinite(combined_audio_layers)):
                st.error("❌ Errore: l'aggiunta del layer Rumore ha introdotto valori non finiti. Sostituzione con zeri.")
                combined_audio_layers[~np.isfinite(combined_audio_layers)] = 0.0
            st.success("✅ Strato Rumore aggiunto!")

        # --- Applica effetti Glitch se abilitati ---
        if enable_glitch_effect:
            st.info("👾 Applicazione effetti Glitch...")
            combined_audio_layers = audio_gen.apply_glitch_effect(
                combined_audio_layers,
                variation_movement_data,
                glitch_threshold_user,
                glitch_duration_frames_user,
                glitch_intensity_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            # Controllo: Applicazione Glitch
            if not np.all(np.isfinite(combined_audio_layers)):
                st.error("❌ Errore: l'applicazione del Glitch ha introdotto valori non finiti. Sostituzione con zeri.")
                combined_audio_layers[~np.isfinite(combined_audio_layers)] = 0.0
            st.success("✅ Effetti Glitch applicati!")

        # --- Processamento degli effetti dinamici (filtro, pitch, time stretch) ---
        with st.spinner("🎧 Applicazione effetti dinamici all'audio generato..."):
            processed_audio = combined_audio_layers # Inizia con l'audio base/combinato
            
            # Controllo di integrità prima di iniziare la catena di effetti
            if not np.all(np.isfinite(processed_audio)):
                st.error("❌ Errore critico: L'audio combinato contiene valori non finiti prima di applicare gli effetti dinamici. Questo potrebbe indicare un problema con i layer precedenti.")
                processed_audio[~np.isfinite(processed_audio)] = 0.0


            # Applica Pitch Shifting e Time Stretching
            if enable_pitch_time_stretch:
                st.info("⏱️ Applicazione Pitch Shifting e Time Stretching...")
                processed_audio = audio_gen.apply_pitch_time_stretch(
                    processed_audio,
                    brightness_data,
                    detail_data,
                    min_pitch_shift_semitones=min_pitch_shift_semitones,
                    max_pitch_shift_semitones=max_pitch_shift_semitones,
                    min_time_stretch_rate=min_time_stretch_rate,
                    max_time_stretch_rate=max_time_stretch_rate,
                    status_text=audio_status_text # Rimosso progress_bar per Pitch/Time Stretch perché la funzione librosa non lo supporta direttamente
                )
                # Controllo: Pitch/Time Stretch
                if not np.all(np.isfinite(processed_audio)):
                    st.error("❌ Errore: Pitch Shift / Time Stretch ha introdotto valori non finiti. Sostituzione con zeri.")
                    processed_audio[~np.isfinite(processed_audio)] = 0.0
                st.success("✅ Effetti pitch e stretch applicati!")
            else:
                st.info("🎶 Pitch Shifting / Time Stretching disabilitato.")


            # --- Applicazione Effetti Sonori Dinamici unificati ---
            if enable_dynamic_effects:
                # Applica Filtro Avanzato
                st.info("⚙️ Applicazione Filtro Avanzato...")
                processed_audio = audio_gen.apply_dynamic_filter(
                    processed_audio,
                    brightness_data,
                    detail_data,
                    min_cutoff=min_cutoff_adv,
                    max_cutoff=max_cutoff_adv,
                    min_res=min_resonance_adv,
                    max_res=max_resonance_adv,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text,
                    filter_type=filter_type_user
                )
                # Controllo: Filtro Avanzato
                if not np.all(np.isfinite(processed_audio)):
                    st.error("❌ Errore: I filtri avanzati hanno introdotto valori non finiti. Sostituzione con zeri.")
                    processed_audio[~np.isfinite(processed_audio)] = 0.0
                st.success("✅ Filtri avanzati applicati!")

                # Applica Effetto di Modulazione (Chorus/Flanger/Tremolo)
                if modulation_effect_type != "none":
                    st.info(f"🌀 Applicazione Effetto di Modulazione ({modulation_effect_type})...")
                    processed_audio = audio_gen.apply_modulation_effect(
                        processed_audio,
                        variation_movement_data,
                        detail_data,
                        progress_bar=audio_progress_bar,
                        status_text=audio_status_text,
                        effect_type=modulation_effect_type,
                        intensity=modulation_intensity,
                        rate=modulation_rate
                    )
                    # Controllo: Modulazione
                    if not np.all(np.isfinite(processed_audio)):
                        st.error("❌ Errore: L'effetto di modulazione ha introdotto valori non finiti. Sostituzione con zeri.")
                        processed_audio[~np.isfinite(processed_audio)] = 0.0
                    st.success(f"✅ Effetto di Modulazione '{modulation_effect_type}' applicato!")
                else:
                    st.info("🎶 Effetto di Modulazione disabilitato.")


                # Applica Delay
                st.info(" echoes ⏰ Applicazione Delay...")
                processed_audio = audio_gen.apply_delay_effect(
                    processed_audio,
                    movement_data,
                    detail_data,
                    max_delay_time_user,
                    max_delay_feedback_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                # Controllo: Delay
                if not np.all(np.isfinite(processed_audio)):
                    st.error("❌ Errore: L'effetto Delay ha introdotto valori non finiti. Sostituzione con zeri.")
                    processed_audio[~np.isfinite(processed_audio)] = 0.0
                st.success("✅ Effetto Delay applicato!")

                # Applica Riverbero
                st.info("✨ Applicazione Riverbero...")
                processed_audio = audio_gen.apply_reverb_effect(
                    processed_audio,
                    detail_data,
                    brightness_data,
                    max_reverb_decay_user,
                    max_reverb_wet_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                # Controllo: Riverbero
                if not np.all(np.isfinite(processed_audio)):
                    st.error("❌ Errore: L'effetto Riverbero ha introdotto valori non finiti. Sostituzione con zeri.")
                    processed_audio[~np.isfinite(processed_audio)] = 0.0
                st.success("✅ Effetto Riverbero applicato!")
            else:
                st.info("🎶 Effetti Dinamici Avanzati disabilitati.")
            
            # --- Panning Stereo ---
            st.info("↔️ Applicazione Panning Stereo...")
            processed_audio_stereo = audio_gen.apply_stereo_panning(processed_audio, horizontal_center_data)
            # Controllo: Panning Stereo
            if not np.all(np.isfinite(processed_audio_stereo)):
                st.error("❌ Errore: Panning stereo ha introdotto valori non finiti. Sostituzione con zeri.")
                # Per audio stereo, dobbiamo pulire entrambi i canali
                processed_audio_stereo[~np.isfinite(processed_audio_stereo)] = 0.0
            st.success("✅ Panning stereo applicato!")

            # --- Normalizzazione finale ---
            st.info("🔊 Normalizzazione audio finale...")
            final_audio = audio_gen.normalize_audio(processed_audio_stereo)
            # Controllo: Normalizzazione Finale
            if not np.all(np.isfinite(final_audio)):
                st.error("❌ Errore critico: L'audio normalizzato contiene valori non finiti. L'audio finale potrebbe essere corrotto. Sostituzione con zeri.")
                final_audio[~np.isfinite(final_audio)] = 0.0
            st.success("✅ Audio normalizzato!")

        # --- Salvataggio audio ---
        st.info(f"💾 Salvataggio audio in '{audio_output_path}'...")
        try:
            # Assicurati che l'audio sia nel formato corretto per soundfile (mono o stereo float32)
            if final_audio.ndim == 2 and final_audio.shape[1] == 1: # Se è stereo con una sola colonna, rendilo mono
                final_audio = final_audio[:, 0]
            sf.write(audio_output_path, final_audio, AUDIO_SAMPLE_RATE)
            st.success("✅ Audio salvato correttamente!")
            st.audio(audio_output_path, format="audio/wav", start_time=0) # Permette all'utente di ascoltare subito
        except Exception as e:
            st.error(f"❌ Errore durante il salvataggio dell'audio: {e}")
            return

        report_text = f"""
        ### Resoconto Generazione Suono

        **Durata Audio:** {video_duration:.2f} secondi
        **Sample Rate:** {AUDIO_SAMPLE_RATE} Hz
        **FPS Base:** {fps:.2f}

        **Layer Abilitati:**
        - Sintesi Sottrattiva: {'Sì' if enable_subtractive_synthesis else 'No'}
        - Sintesi FM: {'Sì' if enable_fm_synthesis else 'No'}
        - Sintesi Granulare: {'Sì' if enable_granular_synthesis else 'No'}
        - Rumore: {'Sì' if enable_noise_effect else 'No'}
        - Glitch: {'Sì' if enable_glitch_effect else 'No'}

        **Effetti Dinamici Abilitati:**
        - Pitch Shifting / Time Stretching: {'Sì' if enable_pitch_time_stretch else 'No'}
        - Filtro Avanzato: {'Sì' if enable_dynamic_effects else 'No'}
        - Modulazione ({modulation_effect_type.capitalize()}): {'Sì' if enable_dynamic_effects and modulation_effect_type != 'none' else 'No'}
        - Delay: {'Sì' if enable_dynamic_effects else 'No'}
        - Riverbero: {'Sì' if enable_dynamic_effects else 'No'}

        **Percorso File Audio:** `{audio_output_path}`
        """
        st.markdown(report_text)

        # --- Opzioni di Download ---
        st.subheader("Scarica il tuo risultato!")
        with open(audio_output_path, "rb") as file:
            st.download_button(
                label="⬇️ Scarica Solo Audio (.wav)",
                data=file.read(),
                file_name=f"{base_name_output}_{unique_id_audio}_generated_audio.wav",
                mime="audio/wav"
            )

        if download_option == "Video con Audio" and input_mode == "Carica un Video" and video_input_path:
            st.info("🎬 Unione audio e video in corso...")
            video_status_text = st.empty()
            try:
                # Determina la risoluzione di output
                target_width, target_height = width, height # Default: originale
                if output_resolution_choice != "Originale":
                    target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]

                # Comando FFmpeg per unire audio e video e ricodificare a risoluzione desiderata
                command = [
                    FFMPEG_PATH,
                    "-i", video_input_path,
                    "-i", audio_output_path,
                    "-c:v", "libx264", # Codec video
                    "-preset", "fast", # Velocità di codifica (fast, medium, slow)
                    "-crf", "23", # Qualità (0-51, 0 lossless, 23 default, 51 peggiore)
                    "-vf", f"scale={target_width}:{target_height}", # Scala la risoluzione
                    "-c:a", "aac", # Codec audio
                    "-b:a", "192k", # Bitrate audio
                    "-pix_fmt", "yuv420p", # Formato pixel per compatibilità
                    final_video_path
                ]
                
                # Esegui il comando FFmpeg
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Leggi l'output di stderr per il progresso
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if "frame=" in line:
                        video_status_text.text(f"Unione Video: {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    st.success(f"✅ Video con audio salvato in '{final_video_path}'")
                    with open(final_video_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Scarica Video con Audio (.mp4)",
                            data=file.read(),
                            file_name=f"{base_name_output}_{unique_id_audio}_final_videosound.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error(f"❌ Errore durante l'unione del video e dell'audio. FFmpeg Code: {process.returncode}")
                    st.code(process.stderr.read()) # Mostra l'output di errore di FFmpeg
            except Exception as e:
                st.error(f"❌ Errore imprevisto durante l'unione del video: {e}")
        
        # Pulizia dei file temporanei
        if os.path.exists("temp"):
            st.info("🗑️ Pulizia dei file temporanei...")
            try:
                shutil.rmtree("temp")
                st.success("✅ File temporanei rimossi.")
            except Exception as e:
                st.warning(f"⚠️ Impossibile rimuovere i file temporanei: {e}")
        
        # Forza la garbage collection per liberare memoria
        gc.collect()

if __name__ == "__main__":
    main()
