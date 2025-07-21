import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa
import librosa.display
import time # Importa il modulo time

# --- COSTANTI GLOBALI ---
AUDIO_SAMPLE_RATE = 44100 # Hz
AUDIO_FPS = 30 # Frame rate per l'analisi e la generazione audio
MAX_DURATION = 120 # Secondi, durata massima per i video/audio generati (ad esempio, 2 minuti)

FORMAT_RESOLUTIONS = {
    "Originale": (0, 0), # Mantiene la risoluzione originale del video
    "720p (HD)": (1280, 720),
    "1080p (Full HD)": (1920, 1080)
}
# --- FINE COSTANTI GLOBALI ---

# --- FUNZIONI HELPER ---

def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è installato e accessibile nel PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def validate_video_file(uploaded_file) -> bool:
    """Controlla la validità del file video caricato."""
    if uploaded_file.size == 0:
        st.error("❌ Il file video è vuoto. Carica un file valido.")
        return False
    if uploaded_file.type not in ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"]:
        st.error("❌ Formato video non supportato. Carica un file MP4, MOV, AVI o MKV.")
        return False
    # Puoi aggiungere qui ulteriori controlli, ad esempio sulla dimensione massima.
    return True

def analyze_video_frames(video_path: str, progress_bar, status_text) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Analizza i frame di un video per estrarre dati di luminosità, dettaglio/contrasto, movimento e centro di massa orizzontale.
    Restituisce array di luminosità, dettaglio, movimento, variazione di movimento e centro di massa,
    oltre a larghezza, altezza, FPS e durata del video.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"❌ Impossibile aprire il file video: {video_path}")
        return None, None, None, None, None, None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps if fps > 0 else 0

    brightness_data = []
    detail_data = []
    movement_data = []
    variation_movement_data = [] # Per catturare la velocità di cambiamento del movimento
    horizontal_center_data = [] # Per il centro di massa orizzontale

    prev_gray_frame = None
    prev_movement = 0.0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Luminosità (media dei pixel, normalizzata a 0-1)
        brightness = np.mean(gray_frame) / 255.0
        brightness_data.append(brightness)

        # Dettaglio/Contrasto (deviazione standard dei pixel, normalizzata a 0-1)
        detail = np.std(gray_frame) / 255.0
        detail_data.append(detail)

        # Movimento (differenza assoluta tra frame consecutivi, normalizzata a 0-1)
        current_movement = 0.0
        if prev_gray_frame is not None:
            current_movement = np.sum(cv2.absdiff(gray_frame, prev_gray_frame)) / (gray_frame.size * 255.0)
            movement_data.append(current_movement)
        else:
            movement_data.append(0.0) # Primo frame ha movimento 0

        # Variazione del movimento (quanto velocemente cambia il movimento)
        variation_movement = abs(current_movement - prev_movement)
        variation_movement_data.append(variation_movement)
        
        # Centro di massa orizzontale
        if np.sum(gray_frame) > 0: # Evita divisione per zero per frame completamente neri
            moments = cv2.moments(gray_frame)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                normalized_cx = cx / width # Normalizzato tra 0 (sinistra) e 1 (destra)
                horizontal_center_data.append(normalized_cx)
            else:
                horizontal_center_data.append(0.5) # Centro se il frame è nero
        else:
            horizontal_center_data.append(0.5) # Centro se il frame è nero

        prev_gray_frame = gray_frame
        prev_movement = current_movement

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"📊 Analisi Frame {frame_count}/{total_frames} | Lum: {brightness:.2f} | Det: {detail:.2f} | Mov: {movement_data[-1]:.2f} | VarMov: {variation_movement_data[-1]:.2f} | Pan: {horizontal_center_data[-1]:.2f}")

    cap.release()
    gc.collect() # Libera memoria

    # Assicurati che tutti gli array abbiano la stessa lunghezza
    # A volte movement_data e variation_movement_data sono più corti di 1
    max_len = len(brightness_data)
    for arr in [movement_data, variation_movement_data, horizontal_center_data]:
        if len(arr) < max_len:
            while len(arr) < max_len:
                # Ripeti l'ultimo valore o un valore predefinito
                arr.append(arr[-1] if len(arr) > 0 else 0.5 if arr is horizontal_center_data else 0.0)

    return np.array(brightness_data), np.array(detail_data), np.array(movement_data), np.array(variation_movement_data), np.array(horizontal_center_data), width, height, fps, video_duration

# --- CLASSE AUDIO GENERATOR ---

class AudioGenerator:
    def __init__(self, sample_rate: int, video_fps: int):
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.fade_duration = 0.1 # Secondi per fade in/out per evitare click

    def _map_value(self, value, old_min, old_max, new_min, new_max):
        """Mappa un valore da un intervallo a un altro."""
        if old_max == old_min: # Evita divisione per zero
            return new_min
        mapped_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return np.clip(mapped_value, new_min, new_max)

    def _create_envelope(self, num_samples: int, attack_time: float, release_time: float) -> np.ndarray:
        """Crea un inviluppo ADSR semplice (solo A e R)."""
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        envelope = np.ones(num_samples, dtype=np.float32)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        return envelope

    def generate_subtractive_waveform(self, num_samples: int, waveform_type: str = "sawtooth") -> np.ndarray:
        """Genera un'onda base per la sintesi sottrattiva."""
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False)
        freq = 220 # Frequenza fissa di base, sarà modulata dal filtro

        if waveform_type == "sine":
            waveform = np.sin(2 * np.pi * freq * t)
        elif waveform_type == "square":
            waveform = np.sign(np.sin(2 * np.pi * freq * t))
        elif waveform_type == "sawtooth":
            waveform = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif waveform_type == "triangle":
            waveform = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        else:
            waveform = np.zeros(num_samples) # Fallback

        # Applica un inviluppo per evitare click all'inizio e alla fine
        fade_samples = int(self.fade_duration * self.sample_rate)
        waveform[:fade_samples] *= np.linspace(0, 1, fade_samples)
        waveform[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return waveform.astype(np.float32)

    def apply_dynamic_filter(self, audio_data: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray,
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float,
                             progress_bar, status_text, filter_type: str = "lowpass") -> np.ndarray:
        """Applica un filtro dinamico (lowpass, highpass, bandpass, bandstop) modulato dai dati."""
        filtered_audio = np.copy(audio_data)
        
        # Mappa i dati di analisi al frame rate audio
        brightness_interp = np.interp(np.arange(len(audio_data)),
                                      np.arange(len(brightness_data)) * (self.sample_rate / self.video_fps),
                                      brightness_data)
        detail_interp = np.interp(np.arange(len(audio_data)),
                                  np.arange(len(detail_data)) * (self.sample_rate / self.video_fps),
                                  detail_data)

        block_size = self.sample_rate // self.video_fps # Applica il filtro per blocchi
        
        for i in range(0, len(audio_data), block_size):
            end_idx = min(i + block_size, len(audio_data))
            current_brightness = brightness_interp[i]
            current_detail = detail_interp[i]

            # Mappa luminosità al cutoff
            cutoff_freq = self._map_value(current_brightness, 0.0, 1.0, min_cutoff, max_cutoff)
            # Mappa dettaglio alla risonanza (Q)
            resonance_q = self._map_value(current_detail, 0.0, 1.0, min_res, max_res)

            # Progetta il filtro
            # 'Q' in scipy è 1/BW, mentre la risonanza audio è spesso Q. Quindi usiamo 1/Q per il BW.
            if filter_type == "lowpass":
                b, a = butter(2, cutoff_freq, btype='low', fs=self.sample_rate, analog=False, output='ba')
            elif filter_type == "highpass":
                b, a = butter(2, cutoff_freq, btype='high', fs=self.sample_rate, analog=False, output='ba')
            elif filter_type == "bandpass":
                nyq = 0.5 * self.sample_rate
                low_cut = max(20, cutoff_freq - resonance_q * 10) # Semplificazione per banda
                high_cut = min(self.sample_rate / 2 - 100, cutoff_freq + resonance_q * 10)
                b, a = butter(2, [low_cut / nyq, high_cut / nyq], btype='band', analog=False, output='ba')
            elif filter_type == "bandstop":
                nyq = 0.5 * self.sample_rate
                low_cut = max(20, cutoff_freq - resonance_q * 10) # Semplificazione per banda
                high_cut = min(self.sample_rate / 2 - 100, cutoff_freq + resonance_q * 10)
                b, a = butter(2, [low_cut / nyq, high_cut / nyq], btype='bandstop', analog=False, output='ba')
            else: # Default a lowpass
                b, a = butter(2, cutoff_freq, btype='low', fs=self.sample_rate, analog=False, output='ba')
            
            # Applica il filtro al blocco corrente
            filtered_audio[i:end_idx] = lfilter(b, a, audio_data[i:end_idx])

            progress_bar.progress(end_idx / len(audio_data))
            status_text.text(f"🎧 Applicazione Filtro Dinamico... Freq: {cutoff_freq:.0f} Hz, Res: {resonance_q:.1f}")

        return filtered_audio

    def generate_fm_layer(self, num_samples: int, brightness_data: np.ndarray, movement_data: np.ndarray,
                          min_carrier_freq: float, max_carrier_freq: float,
                          min_modulator_freq: float, max_modulator_freq: float,
                          min_mod_index: float, max_mod_index: float,
                          progress_bar, status_text) -> np.ndarray:
        """Genera un layer audio usando la sintesi FM, modulata da luminosità, movimento e dettaglio."""
        fm_layer = np.zeros(num_samples, dtype=np.float32)
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False)

        # Interpola i dati di controllo per allinearli ai campioni audio
        brightness_interp = np.interp(t, np.arange(len(brightness_data)) / self.video_fps, brightness_data)
        movement_interp = np.interp(t, np.arange(len(movement_data)) / self.video_fps, movement_data)
        # Assumiamo che il dettaglio_data sia lo stesso di movement_data per l'indice di modulazione per semplicità qui,
        # ma potresti passare un dettaglio_data separato se lo desideri.
        detail_interp = movement_interp # Puoi usare detail_data effettivo se lo hai

        for i in range(num_samples):
            # Modula la frequenza portante (carrier) con la luminosità
            carrier_freq = self._map_value(brightness_interp[i], 0.0, 1.0, min_carrier_freq, max_carrier_freq)
            
            # Modula la frequenza modulante con il movimento
            modulator_freq = self._map_value(movement_interp[i], 0.0, 1.0, min_modulator_freq, max_modulator_freq)
            
            # Modula l'indice di modulazione con il dettaglio
            mod_index = self._map_value(detail_interp[i], 0.0, 1.0, min_mod_index, max_mod_index)

            # Calcola il valore del campione FM
            fm_layer[i] = np.sin(2 * np.pi * carrier_freq * t[i] + mod_index * np.sin(2 * np.pi * modulator_freq * t[i]))

            if i % (self.sample_rate * 2) == 0: # Aggiorna ogni 2 secondi per performance
                progress_bar.progress(i / num_samples)
                status_text.text(f"🎵 Generazione FM... Carr: {carrier_freq:.0f}Hz, Mod: {modulator_freq:.0f}Hz, Index: {mod_index:.1f}")
        
        # Applica inviluppo per fade
        fade_samples = int(self.fade_duration * self.sample_rate)
        fm_layer[:fade_samples] *= np.linspace(0, 1, fade_samples)
        fm_layer[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return fm_layer.astype(np.float32)

    def generate_granular_layer(self, num_samples: int, brightness_data: np.ndarray, detail_data: np.ndarray, movement_data: np.ndarray,
                                min_grain_freq: float, max_grain_freq: float,
                                min_grain_density: float, max_grain_density: float,
                                min_grain_duration: float, max_grain_duration: float,
                                progress_bar, status_text) -> np.ndarray:
        """Genera un layer audio usando la sintesi granulare, modulata dai dati."""
        granular_layer = np.zeros(num_samples, dtype=np.float32)
        
        # Interpola i dati di controllo
        t_control = np.arange(len(brightness_data)) / self.video_fps
        t_audio = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False)

        brightness_interp = np.interp(t_audio, t_control, brightness_data)
        detail_interp = np.interp(t_audio, t_control, detail_data)
        movement_interp = np.interp(t_audio, t_control, movement_data)

        current_time = 0.0
        while current_time < num_samples / self.sample_rate:
            sample_idx = int(current_time * self.sample_rate)
            if sample_idx >= num_samples:
                break

            # Mappa i parametri per il grano corrente
            br = brightness_interp[sample_idx]
            det = detail_interp[sample_idx]
            mov = movement_interp[sample_idx]

            grain_freq = self._map_value(br, 0.0, 1.0, min_grain_freq, max_grain_freq)
            grain_density = self._map_value(mov, 0.0, 1.0, min_grain_density, max_grain_density)
            grain_duration = self._map_value(det, 0.0, 1.0, min_grain_duration, max_grain_duration)

            grain_samples = int(grain_duration * self.sample_rate)
            
            # Genera un piccolo "grano" (es. rumore o onda impulsiva)
            grain = np.random.randn(grain_samples) * 0.1 # Rumore bianco come grano
            # grain = np.sin(2 * np.pi * grain_freq * np.linspace(0, grain_duration, grain_samples, endpoint=False)) * 0.2

            # Applica un piccolo inviluppo al grano per evitare click
            fade_grain_samples = min(int(0.01 * self.sample_rate), grain_samples // 2)
            if fade_grain_samples > 0:
                grain[:fade_grain_samples] *= np.linspace(0, 1, fade_grain_samples)
                grain[-fade_grain_samples:] *= np.linspace(1, 0, fade_grain_samples)
            
            # Aggiungi il grano all'audio finale
            end_grain_idx = min(sample_idx + grain_samples, num_samples)
            granular_layer[sample_idx:end_grain_idx] += grain[:end_grain_idx - sample_idx]

            # Avanza il tempo in base alla densità
            # Minore densità = maggiore salto di tempo tra i grani
            time_advance = 1.0 / (grain_density * self.video_fps + 0.1) # Aggiungi 0.1 per evitare divisione per zero
            current_time += time_advance

            if sample_idx % (self.sample_rate * 2) == 0:
                progress_bar.progress(sample_idx / num_samples)
                status_text.text(f"🎵 Generazione Granulare... Freq: {grain_freq:.1f}Hz, Dens: {grain_density:.1f}, Dur: {grain_duration:.3f}s")
        
        # Applica inviluppo finale
        fade_samples = int(self.fade_duration * self.sample_rate)
        granular_layer[:fade_samples] *= np.linspace(0, 1, fade_samples)
        granular_layer[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return granular_layer.astype(np.float32)

    def add_noise_layer(self, audio_data: np.ndarray, detail_data: np.ndarray,
                        min_amp: float, max_amp: float,
                        progress_bar, status_text) -> np.ndarray:
        """Aggiunge un layer di rumore bianco modulato dal dettaglio."""
        noise_layer = np.random.randn(len(audio_data)).astype(np.float32)
        
        detail_interp = np.interp(np.arange(len(audio_data)),
                                  np.arange(len(detail_data)) * (self.sample_rate / self.video_fps),
                                  detail_data)
        
        modulated_noise = np.zeros_like(noise_layer)
        for i in range(len(noise_layer)):
            current_amp = self._map_value(detail_interp[i], 0.0, 1.0, min_amp, max_amp)
            modulated_noise[i] = noise_layer[i] * current_amp
            if i % (self.sample_rate * 2) == 0:
                progress_bar.progress(i / len(audio_data))
                status_text.text(f"🎵 Aggiunta Rumore... Ampiezza: {current_amp:.2f}")

        return audio_data + modulated_noise

    def apply_glitch_effect(self, audio_data: np.ndarray, variation_movement_data: np.ndarray,
                            glitch_threshold: float, glitch_duration_frames: int, glitch_intensity: float,
                            progress_bar, status_text) -> np.ndarray:
        """Applica effetti glitch in base alla variazione del movimento."""
        glitched_audio = np.copy(audio_data)
        
        variation_movement_interp = np.interp(np.arange(len(audio_data)),
                                              np.arange(len(variation_movement_data)) * (self.sample_rate / self.video_fps),
                                              variation_movement_data)
        
        glitch_duration_samples = int(glitch_duration_frames * (self.sample_rate / self.video_fps)) # Durata in campioni
        
        for i in range(len(glitched_audio)):
            if variation_movement_interp[i] > glitch_threshold:
                # Applica un glitch: es. salto, ripetizione o inversione di un piccolo segmento
                start_glitch = max(0, i - glitch_duration_samples // 2)
                end_glitch = min(len(glitched_audio), i + glitch_duration_samples // 2)
                
                if end_glitch > start_glitch:
                    segment = glitched_audio[start_glitch:end_glitch]
                    
                    # Esempio di glitch: inversione e amplificazione
                    glitched_segment = segment[::-1] * (1.0 + glitch_intensity)
                    
                    glitched_audio[start_glitch:end_glitch] = glitched_segment
                
                # Per evitare glitch troppo ravvicinati
                i += glitch_duration_samples # Salta in avanti dopo un glitch
            
            if i % (self.sample_rate * 2) == 0:
                progress_bar.progress(i / len(glitched_audio))
                status_text.text(f"⚙️ Applicazione Glitch... VarMov: {variation_movement_interp[i]:.2f}")

        return glitched_audio

    def apply_pitch_time_stretch(self, audio_data: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray,
                                 min_pitch_shift_semitones: float, max_pitch_shift_semitones: float,
                                 min_time_stretch_rate: float, max_time_stretch_rate: float,
                                 progress_bar, status_text) -> np.ndarray:
        """Applica pitch shifting e time stretching modulati dai dati."""
        # Librosa richiede l'intero segnale per queste operazioni, non è per-sample
        # Applicheremo la trasformazione una volta sola con valori medi o all'inizio/fine
        # Questo è un compromesso per l'efficienza in Streamlit.
        # Per una vera modulazione dinamica per-sample, sarebbe molto più complesso.

        # Calcola i valori medi per l'intera traccia (o puoi prendere i valori iniziali/finali)
        avg_brightness = np.mean(brightness_data)
        avg_detail = np.mean(detail_data)

        pitch_shift_semitones = self._map_value(avg_brightness, 0.0, 1.0, min_pitch_shift_semitones, max_pitch_shift_semitones)
        time_stretch_rate = self._map_value(avg_detail, 0.0, 1.0, min_time_stretch_rate, max_time_stretch_rate)

        st.info(f"🎶 Applicando Pitch Shift di {pitch_shift_semitones:.2f} semitoni e Time Stretch di fattore {time_stretch_rate:.2f}...")

        # Pitch shifting
        y_pitched = librosa.effects.pitch_shift(
            y=audio_data, sr=self.sample_rate, n_steps=pitch_shift_semitones
        )
        
        # Time stretching
        # Assicurati che il rate non sia troppo piccolo o grande
        time_stretch_rate = np.clip(time_stretch_rate, 0.1, 10.0) # Limita per evitare problemi
        # librosa.effects.time_stretch può introdurre artefatti o cambiare la lunghezza
        # Se si cambia la lunghezza, l'audio deve essere riallineato agli altri layer, il che è complesso.
        # Per ora, usiamo librosa.effects.time_stretch, ma sii consapevole degli artefatti.
        y_stretched = librosa.effects.time_stretch(
            y=y_pitched, rate=time_stretch_rate, n_fft=2048, hop_length=512
        )
        
        # Riapplicare la lunghezza originale se time_stretch ha cambiato la lunghezza
        if len(y_stretched) != len(audio_data):
            # Tronca o pad with zeros
            if len(y_stretched) > len(audio_data):
                y_stretched = y_stretched[:len(audio_data)]
            else:
                y_stretched = np.pad(y_stretched, (0, len(audio_data) - len(y_stretched)), 'constant')

        progress_bar.progress(1.0)
        status_text.text("✅ Pitch Shift e Time Stretch applicati!")
        return y_stretched.astype(np.float32)

    def apply_modulation_effect(self, audio_data: np.ndarray, variation_movement_data: np.ndarray, detail_data: np.ndarray,
                                progress_bar, status_text, effect_type: str = "chorus", intensity: float = 0.5, rate: float = 0.1) -> np.ndarray:
        """Applica un effetto di modulazione (chorus, flanger, phaser) modulato dai dati."""
        modulated_audio = np.copy(audio_data)

        # La modulazione è un effetto globale, non per-sample con librosa facilmente.
        # Usiamo l'intensità e il rate forniti dall'utente.
        # I dati visivi potrebbero modulare questi parametri nel tempo, ma per la prima versione
        # li useremo per influenzare l'esistenza/intensità dell'effetto.

        # Potresti mappare `intensity` con `detail_data` e `rate` con `variation_movement_data`
        # per una modulazione più dinamica, ma richiederebbe un'implementazione a blocchi o un DSP più avanzato.
        # Per ora, useremo i valori fissi o medi.
        
        # Librosa non ha effetti di modulazione diretti come chorus/flanger/phaser.
        # Questa è una semplificazione o un placeholder.
        # Per una vera implementazione, sarebbero necessarie librerie DSP più complete (es. Pytecs, Pysounddevice con elaborazione real-time).
        
        # Per semplicità, qui faremo una modulazione di ampiezza basata su detail_data e movement_data
        # come un effetto "di modulazione" generico, che può essere interpretato come un tremolo o auto-wah base.

        detail_interp = np.interp(np.arange(len(audio_data)),
                                  np.arange(len(detail_data)) * (self.sample_rate / self.video_fps),
                                  detail_data)
        
        variation_movement_interp = np.interp(np.arange(len(audio_data)),
                                              np.arange(len(variation_movement_data)) * (self.sample_rate / self.video_fps),
                                              variation_movement_data)

        if effect_type == "tremolo":
            # Modulazione di ampiezza
            lfo_freq = self._map_value(variation_movement_interp, 0.0, 1.0, 0.1, 10.0) # Frequenza LFO da var. movimento
            tremolo_depth = self._map_value(detail_interp, 0.0, 1.0, 0.0, intensity) # Profondità da dettaglio

            t_lfo = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data), endpoint=False)
            
            for i in range(len(audio_data)):
                current_lfo_freq = lfo_freq[i] if isinstance(lfo_freq, np.ndarray) else lfo_freq
                current_tremolo_depth = tremolo_depth[i] if isinstance(tremolo_depth, np.ndarray) else tremolo_depth
                
                # Modulatore sinusoidale per l'ampiezza
                modulator = 1 + current_tremolo_depth * np.sin(2 * np.pi * current_lfo_freq * t_lfo[i])
                modulated_audio[i] *= modulator
                
            status_text.text(f"🎧 Applicazione Tremolo... Profondità: {intensity:.2f}, Rate: {rate:.2f}")

        elif effect_type in ["chorus", "flanger", "phaser", "none"]:
            # Per questi, librosa non ha implementazioni dirette.
            # Qui si potrebbe aggiungere codice custom DSP o usare una libreria esterna.
            # Per ora, se non è tremolo, non applichiamo nulla o una modulazione base.
            if effect_type != "none":
                st.warning(f"⚠️ L'effetto '{effect_type}' è un placeholder. Librosa non lo supporta direttamente. Verrà applicato un effetto minimo.")
                # Esempio minimo: un leggero delay variabile
                max_delay_s = 0.005  # max 5ms
                delay_samples = int(max_delay_s * self.sample_rate)
                
                delay_line = np.zeros(delay_samples, dtype=np.float32)
                
                for i in range(len(audio_data)):
                    # Modula il tempo di delay con detail_data
                    current_delay_time_norm = self._map_value(detail_interp[i], 0.0, 1.0, 0.0, 1.0)
                    actual_delay_idx = int(current_delay_time_norm * delay_samples)
                    
                    if i - actual_delay_idx >= 0:
                        delay_val = modulated_audio[i - actual_delay_idx]
                        modulated_audio[i] += delay_val * (intensity * 0.5) # Aggiunge il segnale ritardato con un'intensità
            status_text.text(f"🎧 Placeholder per {effect_type} applicato.")
        else:
            st.warning(f"Tipo di effetto di modulazione '{effect_type}' non riconosciuto. Nessun effetto applicato.")
        
        progress_bar.progress(1.0)
        return modulated_audio.astype(np.float32)

    def apply_delay_effect(self, audio_data: np.ndarray, movement_data: np.ndarray, detail_data: np.ndarray,
                           max_delay_time: float, max_delay_feedback: float,
                           progress_bar, status_text) -> np.ndarray:
        """Applica un effetto delay (eco) modulato dai dati."""
        delayed_audio = np.copy(audio_data)
        
        # Interpola i dati di controllo
        movement_interp = np.interp(np.arange(len(audio_data)),
                                    np.arange(len(movement_data)) * (self.sample_rate / self.video_fps),
                                    movement_data)
        detail_interp = np.interp(np.arange(len(audio_data)),
                                  np.arange(len(detail_data)) * (self.sample_rate / self.video_fps),
                                  detail_data)

        # Linea di delay (buffer circolare)
        max_delay_samples = int(max_delay_time * self.sample_rate)
        if max_delay_samples == 0:
            return delayed_audio # Nessun delay se il tempo massimo è 0

        delay_line = np.zeros(max_delay_samples, dtype=np.float32)
        write_idx = 0

        for i in range(len(audio_data)):
            # Modula il tempo di delay con il movimento
            current_delay_time_norm = self._map_value(movement_interp[i], 0.0, 1.0, 0.0, 1.0)
            read_idx = int(write_idx - (current_delay_time_norm * max_delay_samples)) % max_delay_samples
            
            # Modula il feedback con il dettaglio
            current_feedback = self._map_value(detail_interp[i], 0.0, 1.0, 0.0, max_delay_feedback)
            
            # Leggi dal delay line
            delayed_sample = delay_line[read_idx]

            # Aggiungi il sample corrente al delay line con feedback
            delay_line[write_idx] = audio_data[i] + delayed_sample * current_feedback
            
            # Aggiungi il sample ritardato all'output principale
            delayed_audio[i] += delayed_sample * 0.5 # Mix del segnale dry e wet (0.5)

            write_idx = (write_idx + 1) % max_delay_samples

            if i % (self.sample_rate * 2) == 0:
                progress_bar.progress(i / len(audio_data))
                status_text.text(f"⏰ Applicazione Delay... Time: {current_delay_time_norm * max_delay_time:.2f}s, FB: {current_feedback:.2f}")

        return delayed_audio.astype(np.float32)

    def apply_reverb_effect(self, audio_data: np.ndarray, detail_data: np.ndarray, brightness_data: np.ndarray,
                            max_decay_time: float, max_wet_level: float,
                            progress_bar, status_text) -> np.ndarray:
        """Applica un effetto di riverbero modulato dai dati."""
        # Il riverbero è complesso da implementare dinamicamente per-sample senza librerie dedicate.
        # Useremo una semplificazione con un filtro FIR/IIR lungo o una convoluzione con un impulso
        # Questa implementazione userà una serie di delay in parallelo (comb filters) per simulare un riverbero base.

        reverbed_audio = np.copy(audio_data)

        detail_interp = np.interp(np.arange(len(audio_data)),
                                  np.arange(len(detail_data)) * (self.sample_rate / self.video_fps),
                                  detail_data)
        brightness_interp = np.interp(np.arange(len(audio_data)),
                                      np.arange(len(brightness_data)) * (self.sample_rate / self.video_fps),
                                      brightness_data)
        
        # Parametri di riverbero basici (possono essere modulati)
        num_delays = 8 # Numero di linee di delay parallele
        delay_times = [0.023, 0.031, 0.037, 0.041, 0.043, 0.053, 0.061, 0.067] # Tempi di delay in secondi (primi dispari)
        decay_factors = [0.7, 0.65, 0.75, 0.68, 0.72, 0.69, 0.71, 0.73] # Fattori di decadimento per ogni linea

        delay_buffers = [np.zeros(int(dt * self.sample_rate), dtype=np.float32) for dt in delay_times]
        write_indices = [0] * num_delays

        for i in range(len(audio_data)):
            current_wet_level = self._map_value(detail_interp[i], 0.0, 1.0, 0.0, max_wet_level)
            current_decay_mult = self._map_value(brightness_interp[i], 0.0, 1.0, 0.5, 1.0) # Modula il decadimento

            reverb_output_sum = 0.0
            for j in range(num_delays):
                buf_len = len(delay_buffers[j])
                read_idx = (write_indices[j] + 1) % buf_len # Legge il più "vecchio" campione
                
                # Input al delay line: una parte del segnale originale + il feedback
                input_to_delay = audio_data[i] + delay_buffers[j][read_idx] * (decay_factors[j] * current_decay_mult)
                
                delay_buffers[j][write_indices[j]] = input_to_delay
                reverb_output_sum += delay_buffers[j][read_idx]

                write_indices[j] = (write_indices[j] + 1) % buf_len

            reverbed_audio[i] = audio_data[i] + reverb_output_sum * current_wet_level

            if i % (self.sample_rate * 2) == 0:
                progress_bar.progress(i / len(audio_data))
                status_text.text(f"✨ Applicazione Riverbero... Wet: {current_wet_level:.2f}, Decay: {current_decay_mult * max_decay_time:.2f}s")
        
        return reverbed_audio.astype(np.float32)

    def apply_stereo_panning(self, audio_data: np.ndarray, horizontal_center_data: np.ndarray) -> np.ndarray:
        """Applica il panning stereo basato sul centro di massa orizzontale."""
        # Se l'audio è mono, lo convertiamo in stereo
        if audio_data.ndim == 1:
            stereo_audio = np.zeros((len(audio_data), 2), dtype=np.float32)
            stereo_audio[:, 0] = audio_data # Inizialmente entrambi i canali hanno lo stesso audio
            stereo_audio[:, 1] = audio_data
        else:
            stereo_audio = np.copy(audio_data)

        horizontal_center_interp = np.interp(np.arange(len(audio_data)),
                                             np.arange(len(horizontal_center_data)) * (self.sample_rate / self.video_fps),
                                             horizontal_center_data)

        # Modifica l'ampiezza di ciascun canale in base al panning
        for i in range(len(stereo_audio)):
            pan_norm = horizontal_center_interp[i] # 0.0 (sinistra) a 1.0 (destra)
            
            # Metodo di panning a potenza costante (più naturale)
            gain_left = np.sqrt(1 - pan_norm)
            gain_right = np.sqrt(pan_norm)

            stereo_audio[i, 0] *= gain_left
            stereo_audio[i, 1] *= gain_right

        return stereo_audio

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalizza l'audio al livello di picco."""
        max_abs_val = np.max(np.abs(audio_data))
        if max_abs_val > 0:
            return audio_data / max_abs_val
        return audio_data

# --- FUNZIONE PRINCIPALE STREAMLIT ---

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

        st.markdown("---")
        st.markdown("**Andamento Dettaglio/Contrasto (controlla risonanza, modulator FM, noise, ecc.)**")
        start_detail = st.slider("Dettaglio Iniziale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.8, 0.05, key="sd")
        end_detail = st.slider("Dettaglio Finale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.2, 0.05, key="ed")
        detail_data = np.linspace(start_detail, end_detail, num_virtual_frames)

        st.markdown("---")
        st.markdown("**Andamento Movimento (controlla glitch, delay, densità granulare, ecc.)**")
        base_movement = np.linspace(0.1, 0.5, num_virtual_frames) # Movimento di base crescente
        random_movement_factor = st.slider("Fattore di Movimento Casuale", 0.0, 1.0, 0.2, 0.05, help="Aggiunge casualità all'andamento del movimento.")
        movement_data = base_movement + (np.random.rand(num_virtual_frames) - 0.5) * random_movement_factor
        movement_data = np.clip(movement_data, 0.0, 1.0) # Assicura che i valori rimangano tra 0 e 1
        variation_movement_data = np.abs(np.diff(movement_data, prepend=movement_data[0]))
        
        st.markdown("---")
        st.markdown("**Andamento Panning Orizzontale (controlla il bilanciamento stereo)**")
        start_pan = st.slider("Panning Iniziale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="sp")
        end_pan = st.slider("Panning Finale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="ep")
        horizontal_center_data = np.linspace(start_pan, end_pan, num_virtual_frames)

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
            ("none", "tremolo", "chorus", "flanger", "phaser"), # Chorus/Flanger/Phaser sono placeholder
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
            st.success("✅ Strato Sintesi Sottrattiva generato!")
        else:
            combined_audio_layers = np.zeros(total_samples, dtype=np.float32) # Assicura che sia inizializzato
            st.info("🎵 Sintesi Sottrattiva disabilitata.")


        # --- Aggiungi lo strato FM se abilitato ---
        if enable_fm_synthesis:
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
            combined_audio_layers += fm_layer * 0.5 # Aggiunge con un certo mix
            st.success("✅ Strato FM generato e combinato!")

        # --- Aggiungi lo strato Granulare se abilitato ---
        if enable_granular_synthesis:
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
            combined_audio_layers += granular_layer * 0.5 # Aggiunge con un certo mix
            st.success("✅ Strato Granulare generato e combinato!")

        # --- Aggiungi il layer di Rumore se abilitato ---
        if enable_noise_effect:
            combined_audio_layers = audio_gen.add_noise_layer(
                combined_audio_layers,
                detail_data,
                min_noise_amp_user,
                max_noise_amp_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            st.success("✅ Strato Rumore aggiunto!")

        # --- Applica effetti Glitch se abilitati ---
        if enable_glitch_effect:
            combined_audio_layers = audio_gen.apply_glitch_effect(
                combined_audio_layers,
                variation_movement_data,
                glitch_threshold_user,
                glitch_duration_frames_user,
                glitch_intensity_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            st.success("✅ Effetti Glitch applicati!")

        # --- Processamento degli effetti dinamici (filtro, pitch, time stretch) ---
        with st.spinner("🎧 Applicazione effetti dinamici all'audio generato..."):
            processed_audio = combined_audio_layers # Inizia con l'audio base/combinato

            # Applica Pitch Shifting e Time Stretching
            if enable_pitch_time_stretch:
                processed_audio = audio_gen.apply_pitch_time_stretch(
                    processed_audio,
                    brightness_data,
                    detail_data,
                    min_pitch_shift_semitones=min_pitch_shift_semitones,
                    max_pitch_shift_semitones=max_pitch_shift_semitones,
                    min_time_stretch_rate=min_time_stretch_rate,
                    max_time_stretch_rate=max_time_stretch_rate,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetti pitch e stretch applicati!")
            else:
                st.info("🎶 Pitch Shifting / Time Stretching disabilitato.")


            # --- Applicazione Effetti Sonori Dinamici unificati ---
            if enable_dynamic_effects:
                # Applica Filtro Avanzato
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
                st.success("✅ Filtri avanzati applicati!")

                # Applica Effetto di Modulazione (Chorus/Flanger/Phaser)
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
                st.success(f"✅ Effetto di Modulazione '{modulation_effect_type}' applicato!")

                # Applica Delay
                processed_audio = audio_gen.apply_delay_effect(
                    processed_audio,
                    movement_data,
                    detail_data,
                    max_delay_time_user,
                    max_delay_feedback_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetto Delay applicato!")

                # Applica Riverbero
                processed_audio = audio_gen.apply_reverb_effect(
                    processed_audio,
                    detail_data,
                    brightness_data,
                    max_reverb_decay_user,
                    max_reverb_wet_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetto Riverbero applicato!")
            else:
                st.info("🎶 Effetti Dinamici Avanzati disabilitati.")
            
            # --- Panning Stereo ---
            processed_audio_stereo = audio_gen.apply_stereo_panning(processed_audio, horizontal_center_data)
            st.success("✅ Panning stereo applicato!")

            # --- Normalizzazione finale ---
            final_audio = audio_gen.normalize_audio(processed_audio_stereo)
            st.success("✅ Audio normalizzato!")

        # --- Salvataggio audio ---
        st.info(f"💾 Salvataggio audio in '{audio_output_path}'...")
        try:
            sf.write(audio_output_path, final_audio, AUDIO_SAMPLE_RATE)
            st.success("✅ Audio salvato correttamente!")
            st.audio(audio_output_path, format="audio/wav", start_time=0) # Permette all'utente di ascoltare subito
        except Exception as e:
            st.error(f"❌ Errore durante il salvataggio dell'audio: {e}")
            return

        # --- Inizio Sezione Resoconto di Generazione del Suono ---
        # Costruisci il testo del resoconto in modo programmatico
        report_text = """
---
## 📝 Resoconto di Generazione del Suono
Il suono che hai appena generato è il risultato di un'interazione dinamica tra le impostazioni che hai scelto e i dati estratti dal video (o definiti manualmente).
Questo soundscape è stato creato con **VideoSound Gen by Loop507**.

### 🚀 Riepilogo Generale
- **Sorgente Input:** `{input_mode}`
""".format(input_mode=input_mode)

        if input_mode == "Carica un Video" and uploaded_file is not None:
            report_text += f"- **Video Originale:** `{uploaded_file.name}`\n"
        
        report_text += f"""
- **Durata Audio Generato:** `{video_duration:.2f}` secondi
- **Frequenza di Campionamento Audio:** `{AUDIO_SAMPLE_RATE}` Hz
- **Frame Rate di Analisi/Virtuale:** `{fps:.2f}` FPS

### 🎶 Effetti e Modulazioni Applicati
"""

        if enable_subtractive_synthesis:
            report_text += f"""
##### Sintesi Sottrattiva (Suono Base)
- **Forma d'Onda:** `{waveform_type_user}`
- **Frequenza di Taglio Filtro:** Modulata dalla **Luminosità** del video/input, variando tra `{min_cutoff_user}` Hz (luminosità bassa) e `{max_cutoff_user}` Hz (luminosità alta).
- **Risonanza Filtro:** Modulata dal **Dettaglio/Contrasto** del video/input, variando tra `{min_resonance_user:.1f}` Q (dettaglio basso) e `{max_resonance_user:.1f}` Q (dettaglio alto).
* Andamento Luminosità Media: `{np.mean(brightness_data):.2f}`. Andamento Dettaglio Medio: `{np.mean(detail_data):.2f}`.
* *Note:* Un aumento della luminosità tenderà ad aprire il filtro rendendo il suono più brillante. Un maggiore dettaglio aumenterà la risonanza, rendendo il suono più "aggressivo" o "tagliente".
"""

        if enable_fm_synthesis:
            report_text += f"""
##### Sintesi FM (Frequenza Modulata)
- **Frequenza Carrier:** Modulata dalla **Luminosità**, variando tra `{min_carrier_freq_user}` Hz e `{max_carrier_freq_user}` Hz.
- **Frequenza Modulatore:** Modulata dal **Movimento**, variando tra `{min_modulator_freq_user}` Hz e `{max_modulator_freq_user}` Hz.
- **Indice di Modulazione:** Modulato dal **Dettaglio/Contrasto**, variando tra `{min_mod_index_user:.1f}` e `{max_mod_index_user:.1f}`.
* Andamento Movimento Medio: `{np.mean(movement_data):.2f}`. Andamento Dettaglio Medio: `{np.mean(detail_data):.2f}`.
* *Note:* L'FM crea timbri complessi e spesso metallici. Maggiore movimento o dettaglio amplifica la modulazione, producendo suoni più ricchi e dissonanti.
"""

        if enable_granular_synthesis:
            report_text += f"""
##### Sintesi Granulare
- **Frequenza Grani:** Modulata dalla **Luminosità** (min: `{min_grain_freq}` Hz, max: `{max_grain_freq}` Hz).
- **Densità Grani:** Modulata dal **Movimento** (min: `{min_grain_density:.1f}`, max: `{max_grain_density:.1f}`).
- **Durata Grani:** Modulata dal **Dettaglio/Contrasto** (min: `{min_grain_duration:.3f}`s, max: `{max_grain_duration:.3f}`s).
* *Note:* La sintesi granulare 'scompone' il suono in piccole particelle (grani). Maggiore movimento e luminosità tendono a creare una texture più densa e rapida.
"""

        if enable_noise_effect:
            report_text += f"""
##### Effetto Rumore
- **Ampiezza Rumore:** Modulata dal **Dettaglio/Contrasto**, variando tra `{min_noise_amp_user:.2f}` e `{max_noise_amp_user:.2f}`.
* *Note:* L'aggiunta di rumore può simulare texture sabbiose o atmosferiche. Un maggiore dettaglio porta a un rumore più evidente.
"""

        if enable_glitch_effect:
            report_text += f"""
##### Effetto Glitch
- **Soglia Glitch:** `{glitch_threshold_user:.2f}` (variazione movimento oltre cui scatta il glitch).
- **Durata Glitch:** `{glitch_duration_frames_user}` frame.
- **Intensità Glitch:** `{glitch_intensity_user:.2f}`.
* Andamento Variazione Movimento Media: `{np.mean(variation_movement_data):.2f}`.
* *Note:* I glitch sono attivati da cambiamenti rapidi nel movimento del video/input, creando interruzioni o distorsioni nel suono.
"""
        
        if enable_pitch_time_stretch:
            report_text += f"""
##### Pitch Shifting e Time Stretching
- **Pitch Shift:** Modulato dalla **Luminosità**, variando tra `{min_pitch_shift_semitones}` e `{max_pitch_shift_semitones}` semitoni.
- **Time Stretch:** Modulato dal **Dettaglio/Contrasto**, variando tra `{min_time_stretch_rate:.2f}` (più lento) e `{max_time_stretch_rate:.2f}` (più veloce).
* *Note:* La luminosità può alzare o abbassare l'intonazione, mentre il dettaglio può rallentare o accelerare il tempo, alterando la percezione della durata.
"""

        if enable_dynamic_effects:
            report_text += f"""
##### Effetti Dinamici Avanzati
- **Filtro Avanzato:** `Abilitato` ({filter_type_user})
* Cutoff da **Luminosità** ({min_cutoff_adv} Hz - {max_cutoff_adv} Hz). Risonanza da **Dettaglio** ({min_resonance_adv:.1f} Q - {max_resonance_adv:.1f} Q).
- **Effetto di Modulazione:** `{modulation_effect_type}` (Intensità: `{modulation_intensity:.2f}`, Rate: `{modulation_rate:.2f}` Hz)
* Modulato da **Variazione Movimento** e **Dettaglio**.
- **Delay:** `Abilitato` (Max Tempo: `{max_delay_time_user:.2f}` s, Max Feedback: `{max_delay_feedback_user:.2f}`)
* Modulato da **Movimento** e **Dettaglio**.
- **Riverbero:** `Abilitato` (Max Decay: `{max_reverb_decay_user:.2f}` s, Max Wet: `{max_reverb_wet_user:.2f}`)
* Modulato da **Dettaglio** e **Luminosità**.
"""

        report_text += f"""
##### Panning Stereo
- L'audio è stato panoramizzato tra sinistra e destra in base alla posizione del **Centro di Massa Orizzontale** del video/input.
* Andamento Centro Orizzontale Medio: `{np.mean(horizontal_center_data):.2f}` (0.0=sinistra, 1.0=destra).

---
Speriamo che questo resoconto ti aiuti a comprendere meglio il processo creativo e il legame tra le immagini (o i tuoi parametri) e il suono generato!
---
"""
        # Ora passa la stringa completa a st.markdown
        st.markdown(report_text)

        # --- Sezione Download ---
        if download_option == "Solo Audio":
            with open(audio_output_path, "rb") as f:
                st.download_button(
                    "⬇️ Scarica Solo Audio (WAV)",
                    f,
                    file_name=f"videosound_generato_audio_{base_name_output}.wav",
                    mime="audio/wav"
                )
            # Pulizia file temporanei
            if os.path.exists(audio_output_path):
                os.remove(audio_output_path)
            if video_input_path and os.path.exists(video_input_path): # Pulisci video input solo se esiste
                os.remove(video_input_path)
            st.info("🗑️ File temporanei puliti.")

        elif download_option == "Video con Audio" and input_mode == "Carica un Video": # Questa opzione è disponibile solo con un video caricato
            if check_ffmpeg():
                with st.spinner("🔗 Unione e ricodifica video/audio con FFmpeg (potrebbe richiedere tempo)..."):
                    target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]

                    if output_resolution_choice == "Originale":
                        target_width = width
                        target_height = height

                    vf_complex = f"scale='min({target_width},iw*({target_height}/ih)):min({target_height},ih*({target_width}/iw))',pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black"

                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", video_input_path,
                            "-i", audio_output_path,
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-crf", "23",
                            "-vf", vf_complex,
                            "-c:a", "aac", "-b:a", "192k",
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-shortest",
                            final_video_path
                        ], capture_output=True, check=True)
                        st.success(f"✅ Video finale con audio salvato in '{final_video_path}'")

                        with open(final_video_path, "rb") as f:
                            st.download_button(
                                "⬇️ Scarica il Video con Audio",
                                f,
                                file_name=f"videosound_generato_{base_name_output}_{output_resolution_choice.replace(' ', '_')}.mp4",
                                mime="video/mp4"
                            )

                        for temp_f in [video_input_path, audio_output_path, final_video_path]:
                            if os.path.exists(temp_f):
                                os.remove(temp_f)
                        st.info("🗑️ File temporanei puliti.")

                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Errore FFmpeg durante l'unione/ricodifica: {e.stderr.decode()}")
                        st.code(e.stdout.decode() + e.stderr.decode())
                    except Exception as e:
                        st.error(f"❌ Errore generico durante l'unione/ricodifica: {str(e)}")
            else:
                st.warning(f"⚠️ FFmpeg non trovato. Il video con audio non può essere unito o ricodificato. L'audio generato è disponibile in '{audio_output_path}'.")
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV temporaneo)",
                        f,
                        file_name=f"videosound_generato_audio_{base_name_output}.wav",
                        mime="audio/wav"
                    )
        elif download_option == "Video con Audio" and input_mode == "Genera Suono Senza Video (Parametri Manuali)":
            st.error("❌ Non puoi scaricare un 'Video con Audio' se hai scelto di generare il suono senza caricare un video. Seleziona 'Solo Audio'.")

if __name__ == "__main__":
    main()
