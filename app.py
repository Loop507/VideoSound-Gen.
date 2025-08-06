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
import re
import base64
import json
import zlib
import urllib.parse
import sys

# Costanti globali (puoi modificarle)
MAX_DURATION = 300  # Durata massima del video in secondi
MIN_DURATION = 1.0  # Durata minima del video in secondi
MAX_FILE_SIZE = 50 * 1024 * 1024  # Dimensione massima del file (50 MB)
AUDIO_SAMPLE_RATE = 44100 # Frequenza di campionamento per l'audio generato

# Definizioni delle risoluzioni per i formati
FORMAT_RESOLUTIONS = {
    "Originale": (0, 0),
    "1:1 (Quadrato)": (720, 720),
    "16:9 (Orizzontale)": (1280, 720),
    "9:16 (Verticale)": (720, 1280)
}

# === NUOVE FUNZIONI PER SALVARE E CARICARE I PRESET ===
def save_state_to_string(state: dict) -> str:
    """Serializza lo stato dell'app in una stringa compressa e codificata in base64."""
    # Definisce le chiavi da escludere dal salvataggio (es. dati di file caricati, ecc.)
    keys_to_exclude = ['video_bytes', 'audio_bytes', 'uploaded_file', 'video_placeholder', 'final_video']
    state_to_save = {k: v for k, v in state.items() if k not in keys_to_exclude}
    
    state_json = json.dumps(state_to_save)
    compressed = zlib.compress(state_json.encode('utf-8'))
    encoded = base64.b64encode(compressed)
    return encoded.decode('utf-8')

def load_state_from_string(encoded_state: str) -> Optional[dict]:
    """Deserializza lo stato da una stringa compressa e codificata."""
    try:
        compressed = base64.b64decode(encoded_state)
        state_json = zlib.decompress(compressed).decode('utf-8')
        return json.loads(state_json)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento del preset: {e}")
        return None
# =======================================================

def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è installato e disponibile nel PATH."""
    return shutil.which("ffmpeg") is not None

def validate_video_file(uploaded_file) -> bool:
    """Valida le dimensioni del file video caricato."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ Il file è troppo grande. Dimensione massima consentita: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB.")
        return False
    return True

def analyze_video_frames(video_path: str) -> Tuple[list, list, list, list, list, float, float]:
    """
    Analizza i frame di un video per estrarre dati visivi.

    Args:
        video_path (str): Il percorso del file video da analizzare.

    Returns:
        Tuple[list, list, list, list, list, float, float]: Una tupla contenente liste di dati
        per luminosità, dettaglio, movimento, variazione del movimento, centro di massa orizzontale,
        la durata effettiva del video in secondi, e il frame rate (FPS) del video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Impossibile aprire il video: {video_path}")
        return [], [], [], [], [], 0.0, 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = frame_count / fps

    if duration_seconds > MAX_DURATION:
        st.error(f"❌ Video troppo lungo. Durata massima consentita: {MAX_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], 0.0, 0.0
    if duration_seconds < MIN_DURATION:
        st.error(f"❌ Video troppo corto. Durata minima consentita: {MIN_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], 0.0, 0.0

    luminosity_data = []
    detail_data = [] # Misurato come deviazione standard dell'intensità dei pixel
    movement_data = [] # Differenza assoluta media tra frame consecutivi
    variation_movement_data = [] # Variazione del movimento
    horizontal_mass_center_data = [] # Centro di massa orizzontale per il panning

    prev_gray_frame = None
    prev_movement = 0.0

    st.info("🎬 Analisi dei frame video in corso...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    current_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Luminosità (media intensità dei pixel)
        luminosity = np.mean(gray_frame) / 255.0 # Normalizzato tra 0 e 1
        luminosity_data.append(luminosity)

        # Dettaglio (deviazione standard dell'intensità dei pixel)
        detail = np.std(gray_frame) / 255.0 # Normalizzato tra 0 e 1
        detail_data.append(detail)

        # Movimento (differenza assoluta media tra frame consecutivi)
        current_movement = 0.0
        if prev_gray_frame is not None:
            diff = cv2.absdiff(gray_frame, prev_gray_frame)
            current_movement = np.mean(diff) / 255.0 # Normalizzato tra 0 e 1
        movement_data.append(current_movement)

        # Variazione del movimento (differenza tra movimento corrente e precedente)
        variation_movement_data.append(abs(current_movement - prev_movement))

        # Centro di massa orizzontale (per il panning)
        # Calcola i momenti di ordine 0 e 1 per trovare il centro
        M = cv2.moments(gray_frame)
        if M['m00'] != 0 and np.sum(gray_frame) > 0: # Evita divisione per zero e frame completamente neri
            cx = int(M['m10'] / M['m00'])
            horizontal_mass_center_data.append(cx / frame.shape[1]) # Normalizzato tra 0 e 1
        else:
            horizontal_mass_center_data.append(0.5) # Centro se il frame è vuoto o scuro

        prev_gray_frame = gray_frame
        prev_movement = current_movement

        current_frame_idx += 1
        progress = int((current_frame_idx / frame_count) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Analisi frame: {current_frame_idx}/{frame_count}")

    cap.release()
    st.success("✅ Analisi video completata!")

    # Assicurati che tutti gli array abbiano la stessa lunghezza finale
    max_len = len(luminosity_data)
    for arr in [detail_data, movement_data, variation_movement_data, horizontal_mass_center_data]:
        while len(arr) < max_len:
            arr.append(arr[-1] if arr else 0.0) # Aggiunge l'ultimo valore o 0.0 se vuoto

    gc.collect() # Libera memoria

    return luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data, duration_seconds, fps


class AudioGenerator:
    def __init__(self, sample_rate: int, total_duration_seconds: float):
        self.sample_rate = sample_rate
        self.total_duration_seconds = total_duration_seconds
        self.total_samples = int(self.total_duration_seconds * self.sample_rate)
        self.time_array = np.linspace(0, self.total_duration_seconds, self.total_samples, endpoint=False)
        # Calcola gli indici dei frame rispetto all'array temporale
        self.frame_indices_in_time = np.linspace(0, len(self.time_array) - 1, len(self.time_array) // self.sample_rate + 1, endpoint=True, dtype=int)


    def _interp_data_to_audio_length(self, data_per_frame: list) -> np.ndarray:
        """Interpola i dati per frame alla lunghezza dell'array audio."""
        if len(data_per_frame) == 0:
            return np.zeros(self.total_samples)
        
        original_time_points = np.linspace(0, self.total_duration_seconds, len(data_per_frame), endpoint=True)
        return np.interp(self.time_array, original_time_points, data_per_frame)

    def generate_subtractive_waveform(self, freq_data: list, amp_data: list, waveform_type: str = "sine") -> np.ndarray:
        """Genera una forma d'onda sottrattiva base con frequenza e ampiezza dinamiche."""
        freq_interp = self._interp_data_to_audio_length(freq_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)
        phase_increment = 2 * np.pi * freq_interp / self.sample_rate
        phase = np.cumsum(phase_increment)

        if waveform_type == "sine":
            waveform = np.sin(phase)
        elif waveform_type == "square":
            waveform = np.sign(np.sin(phase))
        elif waveform_type == "sawtooth":
            normalized_phase = (phase / (2 * np.pi)) % 1.0
            waveform = 2 * (normalized_phase - 0.5)
        else: # default to sine
            waveform = np.sin(phase)

        audio = waveform * amp_interp
        return audio

    def generate_fm_layer(self, carrier_freq_data: list, mod_freq_data: list, mod_idx_data: list, amp_data: list) -> np.ndarray:
        """Genera un layer di sintesi FM con parametri dinamici."""
        carrier_freq_interp = self._interp_data_to_audio_length(carrier_freq_data)
        mod_freq_interp = self._interp_data_to_audio_length(mod_freq_data)
        mod_idx_interp = self._interp_data_to_audio_length(mod_idx_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)

        mod_phase_increment = 2 * np.pi * mod_freq_interp / self.sample_rate
        carrier_phase_increment = 2 * np.pi * carrier_freq_interp / self.sample_rate

        mod_phase = np.cumsum(mod_phase_increment)
        carrier_phase = np.cumsum(carrier_phase_increment)

        modulator_signal = np.sin(mod_phase)
        carrier_signal = np.sin(carrier_phase + mod_idx_interp * modulator_signal)

        audio = carrier_signal * amp_interp * 0.5
        return audio

    def generate_granular_layer(self, density_data: list, grain_duration_data: list, amp_data: list) -> np.ndarray:
        """Genera un layer di sintesi granulare."""
        density_interp = self._interp_data_to_audio_length(density_data)
        grain_duration_interp = self._interp_data_to_audio_length(grain_duration_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)
        
        samples_per_virtual_frame = int(self.total_samples / len(density_data)) if len(density_data) > 0 else self.total_samples
        
        for i in range(len(density_data)):
            current_density = density_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else density_interp[-1]
            current_grain_dur_seconds = grain_duration_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else grain_duration_interp[-1]
            current_amp = amp_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else amp_interp[-1]

            num_grains_in_segment = int(current_density)
            
            if num_grains_in_segment == 0:
                continue

            grain_dur_samples = int(current_grain_dur_seconds * self.sample_rate)
            grain_dur_samples = max(10, grain_dur_samples)

            for _ in range(num_grains_in_segment):
                start_sample_segment = i * samples_per_virtual_frame
                end_sample_segment = min((i + 1) * samples_per_virtual_frame, self.total_samples)

                if start_sample_segment >= end_sample_segment - grain_dur_samples:
                    continue
                
                start_grain_sample = np.random.randint(start_sample_segment, end_sample_segment - grain_dur_samples)

                grain_freq = 200 + np.random.rand() * 800
                grain_t = np.arange(grain_dur_samples) / self.sample_rate
                grain_waveform = np.sin(2 * np.pi * grain_freq * grain_t)

                hanning_window = np.hanning(grain_dur_samples)
                grain_with_envelope = grain_waveform * hanning_window * current_amp * 0.1

                end_grain_sample = start_grain_sample + grain_dur_samples
                if end_grain_sample <= self.total_samples:
                    audio[start_grain_sample:end_grain_sample] += grain_with_envelope
                else:
                    audio[start_grain_sample:self.total_samples] += grain_with_envelope[:self.total_samples - start_grain_sample]


        return audio

    def add_noise_layer(self, audio_array: np.ndarray, noise_amp_data: list) -> np.ndarray:
        """Aggiunge un layer di rumore modulato all'audio esistente."""
        noise_amp_interp = self._interp_data_to_audio_length(noise_amp_data)
        noise_layer = np.random.normal(0, 1, self.total_samples) * noise_amp_interp * 0.2
        return audio_array + noise_layer

    def apply_glitch_effect(self, audio_array: np.ndarray, glitch_factor_data: list, glitch_intensity_data: list) -> np.ndarray:
        """Applica un effetto glitch all'audio."""
        glitched_audio = np.copy(audio_array)
        glitched_audio = np.nan_to_num(glitched_audio, nan=0.0)

        glitch_factor_interp = self._interp_data_to_audio_length(glitch_factor_data)
        glitch_intensity_interp = self._interp_data_to_audio_length(glitch_intensity_data)

        glitch_check_interval_samples = int(0.1 * self.sample_rate) # Controlla ogni 100ms
        
        i = 0
        while i < self.total_samples:
            # Assicurati che current_time_idx sia sempre valido per entrambi gli array interpolati
            current_time_idx = min(i, len(glitch_factor_interp) - 1, len(glitch_intensity_interp) - 1)
            
            # Applica il glitch solo se l'indice è valido e la probabilità è soddisfatta
            if current_time_idx >= 0 and np.random.rand() < glitch_factor_interp[current_time_idx]:
                glitch_intensity = glitch_intensity_interp[current_time_idx]
                
                # Durata del glitch basata sull'intensità (minimo 1 campione)
                glitch_duration_samples = int(glitch_intensity * self.sample_rate * 0.05)
                if glitch_duration_samples == 0: glitch_duration_samples = 1
                
                start_glitch_sample = i
                end_glitch_sample = min(start_glitch_sample + glitch_duration_samples, self.total_samples)
                
                if start_glitch_sample < end_glitch_sample:
                    # Definisce la forma (shape) esatta che il segmento glitched_segment deve avere
                    target_slice = glitched_audio[start_glitch_sample:end_glitch_sample]
                    target_segment_shape = target_slice.shape
                    target_segment_length = target_segment_shape[0]

                    # Crea un array vuoto con la forma e il tipo di dati corretti
                    # Questo garantisce che la forma sia sempre compatibile per l'assegnazione finale
                    glitched_segment = np.empty(target_segment_shape, dtype=glitched_audio.dtype)

                    # Estrai il segmento originale su cui applicare il glitch
                    original_segment = glitched_audio[start_glitch_sample:end_glitch_sample]
                    
                    if len(original_segment) == 0:
                        # Se il segmento originale è vuoto (es. un glitch molto corto alla fine dell'audio)
                        # riempi il segmento glitched con zeri.
                        glitched_segment[:] = 0
                    else:
                        # Scegli il tipo di glitch
                        glitch_type = np.random.choice(["repeat", "noise", "reverse"])

                        if glitch_type == "repeat":
                            # Ripeti il segmento originale finché non raggiunge o supera la lunghezza target
                            if original_segment.ndim == 1: # Audio mono
                                num_repeats = int(np.ceil(target_segment_length / len(original_segment)))
                                temp_tiled = np.tile(original_segment, num_repeats)
                                glitched_segment[:] = temp_tiled[:target_segment_length]
                            else: # Audio stereo
                                num_repeats = int(np.ceil(target_segment_length / original_segment.shape[0]))
                                temp_tiled = np.tile(original_segment, (num_repeats, 1))
                                glitched_segment[:, :] = temp_tiled[:target_segment_length, :]

                        elif glitch_type == "noise":
                            # Genera rumore con la forma e il tipo di dati esatti del target
                            glitched_segment[:] = np.random.normal(0, glitch_intensity * 0.5, size=target_segment_shape).astype(glitched_audio.dtype)
                        
                        elif glitch_type == "reverse":
                            # Inverti il segmento originale. La forma rimane la stessa.
                            if original_segment.ndim == 1:
                                glitched_segment[:] = original_segment[::-1]
                            else: # Audio stereo
                                glitched_segment[:] = original_segment[::-1, :]
                    
                    # Assegna il segmento glitched all'array audio principale.
                    # Questa operazione è ora sicura perché glitched_segment ha la forma corretta.
                    glitched_audio[start_glitch_sample:end_glitch_sample] = glitched_segment
                
                # Avanza l'indice 'i' oltre il segmento glitchato
                i = end_glitch_sample
            else:
                # Se non c'è glitch, avanza all'intervallo di controllo successivo
                i += glitch_check_interval_samples

        return glitched_audio

    def apply_delay_effect(self, audio_array: np.ndarray, delay_time_data: list, feedback_data: list) -> np.ndarray:
        """Applica un effetto delay dinamico all'audio."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_array_processed = audio_array

        delayed_audio = np.copy(audio_array_processed)
        
        delay_time_interp = self._interp_data_to_audio_length(delay_time_data)
        feedback_interp = self._interp_data_to_audio_length(feedback_data)

        num_channels = delayed_audio.shape[1]
        
        delay_buffers = [np.zeros(self.sample_rate, dtype=delayed_audio.dtype) for _ in range(num_channels)]
        write_indices = [0] * num_channels

        for i in range(len(delayed_audio)):
            current_delay_time_seconds = np.clip(delay_time_interp[i], 0.01, 0.5)
            current_feedback_gain = np.clip(feedback_interp[i], 0.0, 0.95)

            delay_samples = int(current_delay_time_seconds * self.sample_rate)
            
            for c in range(num_channels):
                read_idx = (write_indices[c] - delay_samples + self.sample_rate) % self.sample_rate
                
                current_sample_channel = delayed_audio[i, c]
                delayed_audio[i, c] += delay_buffers[c][read_idx] * current_feedback_gain
                delay_buffers[c][write_indices[c]] = current_sample_channel + delay_buffers[c][read_idx] * current_feedback_gain

                write_indices[c] = (write_indices[c] + 1) % self.sample_rate

        return delayed_audio.squeeze() if num_channels == 1 else delayed_audio

    def apply_reverb_effect(self, audio_array: np.ndarray, decay_time_data: list, mix_data: list) -> np.ndarray:
        """Applica un semplice effetto di riverbero all'audio."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_array_processed = audio_array

        reverbed_audio = np.copy(audio_array_processed)
        
        decay_time_interp = self._interp_data_to_audio_length(decay_time_data)
        mix_interp = self._interp_data_to_audio_length(mix_data)

        num_channels = reverbed_audio.shape[1]
        num_delay_lines_per_channel = 4 
        
        # Buffer di delay per ogni linea, per ogni canale
        delay_lines = [[np.zeros(self.sample_rate, dtype=reverbed_audio.dtype) for _ in range(num_delay_lines_per_channel)] for _ in range(num_channels)]
        write_indices_reverb = [[0] * num_delay_lines_per_channel for _ in range(num_channels)]
        
        # Tempi di delay fissi per un effetto base di riverbero (in campioni)
        delay_times_samples = [
            int(0.0297 * self.sample_rate),
            int(0.0371 * self.sample_rate),
            int(0.0411 * self.sample_rate),
            int(0.0437 * self.sample_rate)
        ]
        
        for i in range(len(reverbed_audio)):
            current_decay_time = np.clip(decay_time_interp[i], 0.1, 5.0) # decay da 0.1 a 5 secondi
            current_mix = np.clip(mix_interp[i], 0.0, 1.0) # mix da 0 a 1

            for c in range(num_channels):
                dry_signal = audio_array_processed[i, c]
                wet_signal = 0.0

                for dl_idx in range(num_delay_lines_per_channel):
                    delay_samples = delay_times_samples[dl_idx]
                    read_idx = (write_indices_reverb[c][dl_idx] - delay_samples + self.sample_rate) % self.sample_rate
                    
                    feedback_gain = np.exp(-3 * delay_samples / (self.sample_rate * current_decay_time))
                    feedback_gain = np.clip(feedback_gain, 0.0, 0.99) # Assicurati che non superi 1.0 per stabilità

                    delayed_output = delay_lines[c][dl_idx][read_idx]
                    wet_signal += delayed_output

                    delay_lines[c][dl_idx][write_indices_reverb[c][dl_idx]] = dry_signal + delayed_output * feedback_gain
                    
                    write_indices_reverb[c][dl_idx] = (write_indices_reverb[c][dl_idx] + 1) % self.sample_rate

                reverbed_audio[i, c] = dry_signal * (1 - current_mix) + wet_signal * current_mix * 0.2 # Wet attenuato

        return reverbed_audio.squeeze() if num_channels == 1 else reverbed_audio

    def apply_eq_effect(self, audio_array: np.ndarray, low_gain_data: list, mid_gain_data: list, high_gain_data: list) -> np.ndarray:
        """Applica un effetto di equalizzazione dinamica all'audio."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_array_processed = audio_array

        eq_audio = np.copy(audio_array_processed)

        low_gain_interp = self._interp_data_to_audio_length(low_gain_data)
        mid_gain_interp = self._interp_data_to_audio_length(mid_gain_data)
        high_gain_interp = self._interp_data_to_audio_length(high_gain_data)

        nyquist = 0.5 * self.sample_rate

        low_cutoff_freq = 200 / nyquist
        high_cutoff_freq = 2000 / nyquist

        b_low, a_low = butter(2, low_cutoff_freq, btype='low', analog=False)
        b_high, a_high = butter(2, high_cutoff_freq, btype='high', analog=False)

        low_band = lfilter(b_low, a_low, eq_audio, axis=0)
        high_band = lfilter(b_high, a_high, eq_audio, axis=0)

        mid_band = eq_audio - low_band - high_band

        for i in range(len(eq_audio)):
            current_low_gain_db = low_gain_interp[i]
            current_mid_gain_db = mid_gain_interp[i]
            current_high_gain_db = high_gain_interp[i]

            low_gain_linear = 10**(current_low_gain_db / 20)
            mid_gain_linear = 10**(current_mid_gain_db / 20)
            high_gain_linear = 10**(current_high_gain_db / 20)

            for c in range(eq_audio.shape[1]):
                eq_audio[i, c] = (low_band[i, c] * low_gain_linear +
                                  mid_band[i, c] * mid_gain_linear +
                                  high_band[i, c] * high_gain_linear)

        return eq_audio.squeeze() if eq_audio.shape[1] == 1 else eq_audio


def main():
    st.set_page_config(layout="wide", page_title="VideoSound Gen. by Loop507", page_icon="🎵")
    st.markdown("# VideoSound Gen. <small>by Loop507</small>", unsafe_allow_html=True)
    st.markdown("Crea colonne sonore uniche dai tuoi video, trasformando i dati visivi in paesaggi sonori dinamici.")

    # Inizializza session_state con valori di default se non esistono
    for key, val in {
        'duration': 0, 'fps': 30, 'width': 1920, 'height': 1080, 'channels': 2, 'sample_rate': 44100,
        'amplitude': 1000, 'subtractive_on': True, 'sub_freq_src': 'Luminosità', 'sub_amp_src': 'Luminosità',
        'sub_waveform_type': 'sine', 'sub_freq_min': 100, 'sub_freq_max': 800, 'sub_amp_min': 0.1,
        'sub_amp_max': 0.5, 'fm_on': True, 'fm_carr_src': 'Luminosità', 'fm_mod_src': 'Luminosità',
        'fm_idx_src': 'Luminosità', 'fm_amp_src': 'Luminosità', 'fm_carr_min': 200, 'fm_carr_max': 1500,
        'fm_mod_min': 50, 'fm_mod_max': 250, 'fm_idx_min': 0.5, 'fm_idx_max': 5.0, 'fm_amp_min': 0.05,
        'fm_amp_max': 0.3, 'granular_on': True, 'gran_dens_src': 'Dettaglio', 'gran_dur_src': 'Dettaglio',
        'gran_amp_src': 'Dettaglio', 'gran_dens_min': 1, 'gran_dens_max': 5, 'gran_dur_min': 0.02,
        'gran_dur_max': 0.05, 'gran_amp_min': 0.01, 'gran_amp_max': 0.1, 'noise_on': True,
        'noise_amp_src': 'Variazione Movimento', 'noise_amp_min': 0.0, 'noise_amp_max': 0.1,
        'glitch_on': False, 'glitch_factor_src': 'Variazione Movimento', 'glitch_intensity_src': 'Variazione Movimento',
        'glitch_factor_min': 0.01, 'glitch_factor_max': 0.1, 'glitch_intensity_min': 0.1, 'glitch_intensity_max': 0.8,
        'delay_on': False, 'delay_time_src': 'Movimento', 'delay_feedback_src': 'Movimento',
        'delay_time_min': 0.1, 'delay_time_max': 0.3, 'delay_feedback_min': 0.3, 'delay_feedback_max': 0.7,
        'reverb_on': False, 'reverb_decay_src': 'Luminosità', 'reverb_mix_src': 'Luminosità',
        'reverb_decay_min': 1.0, 'reverb_decay_max': 3.0, 'reverb_mix_min': 0.2, 'reverb_mix_max': 0.6,
        'eq_on': False, 'eq_low_src': 'Luminosità', 'eq_mid_src': 'Dettaglio', 'eq_high_src': 'Movimento',
        'eq_gain_min': -10.0, 'eq_gain_max': 10.0, 'output_resolution_choice': '16:9 (Orizzontale)',
        'normalize_audio': True, 'use_original_audio': False, 'original_audio_mix_level': 0.5
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # === GESTIONE INIZIALE DELLO STATO E DEI PARAMETRI URL ===
    query_params = st.query_params
    if 'preset' in query_params:
        encoded_state = query_params['preset']
        loaded_state = load_state_from_string(encoded_state)
        if loaded_state:
            for key, value in loaded_state.items():
                st.session_state[key] = value
            st.info("Preset caricato dall'URL!")
            # Ricarica l'app per applicare i nuovi valori ai widget
            st.experimental_rerun()
    # =======================================================
    
    st.sidebar.header("Carica Video")
    uploaded_file = st.sidebar.file_uploader("Scegli un file video (MP4, MOV, AVI, ecc.)", type=["mp4", "mov", "avi", "mkv"])

    params = {}
    audio_output_path = None
    base_name_output = None

    if uploaded_file is not None:
        if not validate_video_file(uploaded_file):
            if os.path.exists(f"temp_input_{uploaded_file.name}"):
                os.remove(f"temp_input_{uploaded_file.name}")
            return

        st.sidebar.success("✅ Video caricato con successo!")

        video_input_path = f"temp_input_{uploaded_file.name}"
        with open(video_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data, duration_seconds, fps = analyze_video_frames(video_input_path)

        if duration_seconds == 0.0:
            os.remove(video_input_path)
            return

        base_name_output = os.path.splitext(uploaded_file.name)[0]

        st.subheader("Generazione Audio")
        
        audio_generator = AudioGenerator(sample_rate=AUDIO_SAMPLE_RATE, total_duration_seconds=duration_seconds)

        # Scheda per i parametri audio
        tab_sub, tab_fm, tab_gran, tab_noise, tab_fx, tab_eq = st.tabs([
            "Sintesi Sottrattiva", "Sintesi FM", "Sintesi Granulare", "Rumore", "Effetti Audio", "Equalizzatore"
        ])

        with tab_sub:
            st.markdown("### Layer: Sintesi Sottrattiva")
            use_subtractive = st.checkbox("Abilita Sintesi Sottrattiva", value=st.session_state.get('subtractive_on', True), key='subtractive_on')
            params['subtractive_enabled'] = use_subtractive
            if use_subtractive:
                sub_freq_source = st.selectbox("Sorgente Frequenza (Hz)", ["Luminosità", "Dettaglio", "Movimento"], index=["Luminosità", "Dettaglio", "Movimento"].index(st.session_state.get('sub_freq_src', 'Luminosità')), key='sub_freq_src')
                sub_amp_source = st.selectbox("Sorgente Ampiezza (0-1)", ["Luminosità", "Dettaglio", "Movimento"], index=["Luminosità", "Dettaglio", "Movimento"].index(st.session_state.get('sub_amp_src', 'Luminosità')), key='sub_amp_src')
                sub_waveform_type = st.selectbox("Tipo di Forma d'Onda", ["sine", "square", "sawtooth"], index=["sine", "square", "sawtooth"].index(st.session_state.get('sub_waveform_type', 'sine')), key='sub_waveform_type')
                
                sub_freq_min = st.slider("Frequenza Minima (Hz)", 20, 1000, st.session_state.get('sub_freq_min', 100), key='sub_freq_min')
                sub_freq_max = st.slider("Frequenza Massima (Hz)", 20, 1000, st.session_state.get('sub_freq_max', 800), key='sub_freq_max')
                sub_amp_min = st.slider("Ampiezza Minima", 0.0, 1.0, st.session_state.get('sub_amp_min', 0.1), step=0.01, key='sub_amp_min')
                sub_amp_max = st.slider("Ampiezza Massima", 0.0, 1.0, st.session_state.get('sub_amp_max', 0.5), step=0.01, key='sub_amp_max')

                params['sub_freq_source'] = sub_freq_source
                params['sub_amp_source'] = sub_amp_source
                params['sub_waveform_type'] = sub_waveform_type
                params['sub_freq_range'] = (sub_freq_min, sub_freq_max)
                params['sub_amp_range'] = (sub_amp_min, sub_amp_max)

                sub_freq_data_raw = []
                if sub_freq_source == "Luminosità": sub_freq_data_raw = luminosity_data
                elif sub_freq_source == "Dettaglio": sub_freq_data_raw = detail_data
                elif sub_freq_source == "Movimento": sub_freq_data_raw = movement_data

                sub_amp_data_raw = []
                if sub_amp_source == "Luminosità": sub_amp_data_raw = luminosity_data
                elif sub_amp_source == "Dettaglio": sub_amp_data_raw = detail_data
                elif sub_amp_source == "Movimento": sub_amp_data_raw = movement_data
                
                sub_freq_scaled = np.interp(sub_freq_data_raw, (min(sub_freq_data_raw) if sub_freq_data_raw else 0, max(sub_freq_data_raw) if sub_freq_data_raw else 1), (sub_freq_min, sub_freq_max)).tolist()
                sub_amp_scaled = np.interp(sub_amp_data_raw, (min(sub_amp_data_raw) if sub_amp_data_raw else 0, max(sub_amp_data_raw) if sub_amp_data_raw else 1), (sub_amp_min, sub_amp_max)).tolist()
            else:
                sub_freq_scaled = []
                sub_amp_scaled = []

        with tab_fm:
            st.markdown("### Layer: Sintesi FM")
            use_fm = st.checkbox("Abilita Sintesi FM", value=st.session_state.get('fm_on', True), key='fm_on')
            params['fm_enabled'] = use_fm
            if use_fm:
                fm_carrier_source = st.selectbox("Sorgente Frequenza Portante (Hz)", ["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"], index=["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('fm_carr_src', 'Luminosità')), key='fm_carr_src')
                fm_mod_source = st.selectbox("Sorgente Frequenza Modulatore (Hz)", ["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"], index=["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('fm_mod_src', 'Luminosità')), key='fm_mod_src')
                fm_mod_idx_source = st.selectbox("Sorgente Indice Modulazione", ["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"], index=["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('fm_idx_src', 'Luminosità')), key='fm_idx_src')
                fm_amp_source = st.selectbox("Sorgente Ampiezza (0-1)", ["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"], index=["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('fm_amp_src', 'Luminosità')), key='fm_amp_src')

                fm_carrier_min = st.slider("Portante Minima (Hz)", 50, 2000, st.session_state.get('fm_carr_min', 200), key='fm_carr_min')
                fm_carrier_max = st.slider("Portante Massima (Hz)", 50, 2000, st.session_state.get('fm_carr_max', 1500), key='fm_carr_max')
                fm_mod_min = st.slider("Modulatore Minimo (Hz)", 10, 500, st.session_state.get('fm_mod_min', 50), key='fm_mod_min')
                fm_mod_max = st.slider("Modulatore Massimo (Hz)", 10, 500, st.session_state.get('fm_mod_max', 250), key='fm_mod_max')
                fm_mod_idx_min = st.slider("Indice Modulazione Minimo", 0.0, 10.0, st.session_state.get('fm_idx_min', 0.5), step=0.1, key='fm_idx_min')
                fm_mod_idx_max = st.slider("Indice Modulazione Massimo", 0.0, 10.0, st.session_state.get('fm_idx_max', 5.0), step=0.1, key='fm_idx_max')
                fm_amp_min = st.slider("Ampiezza FM Minima", 0.0, 1.0, st.session_state.get('fm_amp_min', 0.05), step=0.01, key='fm_amp_min')
                fm_amp_max = st.slider("Ampiezza FM Massima", 0.0, 1.0, st.session_state.get('fm_amp_max', 0.3), step=0.01, key='fm_amp_max')

                params['fm_carrier_source'] = fm_carrier_source
                params['fm_mod_source'] = fm_mod_source
                params['fm_mod_idx_source'] = fm_mod_idx_source
                params['fm_amp_source'] = fm_amp_source
                params['fm_carrier_range'] = (fm_carrier_min, fm_carrier_max)
                params['fm_mod_range'] = (fm_mod_min, fm_mod_max)
                params['fm_mod_idx_range'] = (fm_mod_idx_min, fm_mod_idx_max)
                params['fm_amp_range'] = (fm_amp_min, fm_amp_max)

                fm_carrier_data_raw = []
                if fm_carrier_source == "Luminosità": fm_carrier_data_raw = luminosity_data
                elif fm_carrier_source == "Dettaglio": fm_carrier_data_raw = detail_data
                elif fm_carrier_source == "Movimento": fm_carrier_data_raw = movement_data
                elif fm_carrier_source == "Variazione Movimento": fm_carrier_data_raw = variation_movement_data

                fm_mod_data_raw = []
                if fm_mod_source == "Luminosità": fm_mod_data_raw = luminosity_data
                elif fm_mod_source == "Dettaglio": fm_mod_data_raw = detail_data
                elif fm_mod_source == "Movimento": fm_mod_data_raw = movement_data
                elif fm_mod_source == "Variazione Movimento": fm_mod_data_raw = variation_movement_data

                fm_mod_idx_data_raw = []
                if fm_mod_idx_source == "Luminosità": fm_mod_idx_data_raw = luminosity_data
                elif fm_mod_idx_source == "Dettaglio": fm_mod_idx_data_raw = detail_data
                elif fm_mod_idx_source == "Movimento": fm_mod_idx_data_raw = movement_data
                elif fm_mod_idx_source == "Variazione Movimento": fm_mod_idx_data_raw = variation_movement_data
                
                fm_amp_data_raw = []
                if fm_amp_source == "Luminosità": fm_amp_data_raw = luminosity_data
                elif fm_amp_source == "Dettaglio": fm_amp_data_raw = detail_data
                elif fm_amp_source == "Movimento": fm_amp_data_raw = movement_data
                elif fm_amp_source == "Variazione Movimento": fm_amp_data_raw = variation_movement_data

                fm_carrier_scaled = np.interp(fm_carrier_data_raw, (min(fm_carrier_data_raw) if fm_carrier_data_raw else 0, max(fm_carrier_data_raw) if fm_carrier_data_raw else 1), (fm_carrier_min, fm_carrier_max)).tolist()
                fm_mod_scaled = np.interp(fm_mod_data_raw, (min(fm_mod_data_raw) if fm_mod_data_raw else 0, max(fm_mod_data_raw) if fm_mod_data_raw else 1), (fm_mod_min, fm_mod_max)).tolist()
                fm_mod_idx_scaled = np.interp(fm_mod_idx_data_raw, (min(fm_mod_idx_data_raw) if fm_mod_idx_data_raw else 0, max(fm_mod_idx_data_raw) if fm_mod_idx_data_raw else 1), (fm_mod_idx_min, fm_mod_idx_max)).tolist()
                fm_amp_scaled = np.interp(fm_amp_data_raw, (min(fm_amp_data_raw) if fm_amp_data_raw else 0, max(fm_amp_data_raw) if fm_amp_data_raw else 1), (fm_amp_min, fm_amp_max)).tolist()
            else:
                fm_carrier_scaled = []
                fm_mod_scaled = []
                fm_mod_idx_scaled = []
                fm_amp_scaled = []

        with tab_gran:
            st.markdown("### Layer: Sintesi Granulare")
            use_granular = st.checkbox("Abilita Sintesi Granulare", value=st.session_state.get('granular_on', True), key='granular_on')
            params['granular_enabled'] = use_granular
            if use_granular:
                gran_density_source = st.selectbox("Sorgente Densità Grani", ["Dettaglio", "Movimento", "Variazione Movimento"], index=["Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('gran_dens_src', 'Dettaglio')), key='gran_dens_src')
                gran_duration_source = st.selectbox("Sorgente Durata Grani (sec)", ["Dettaglio", "Movimento", "Variazione Movimento"], index=["Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('gran_dur_src', 'Dettaglio')), key='gran_dur_src')
                gran_amp_source = st.selectbox("Sorgente Ampiezza Grani (0-1)", ["Dettaglio", "Movimento", "Variazione Movimento"], index=["Dettaglio", "Movimento", "Variazione Movimento"].index(st.session_state.get('gran_amp_src', 'Dettaglio')), key='gran_amp_src')
                
                gran_density_min = st.slider("Densità Minima Grani", 0, 10, st.session_state.get('gran_dens_min', 1), key='gran_dens_min')
                gran_density_max = st.slider("Densità Massima Grani", 0, 10, st.session_state.get('gran_dens_max', 5), key='gran_dens_max')
                gran_duration_min = st.slider("Durata Minima Grani (sec)", 0.01, 0.1, st.session_state.get('gran_dur_min', 0.02), step=0.005, key='gran_dur_min')
                gran_duration_max = st.slider("Durata Massima Grani (sec)", 0.01, 0.1, st.session_state.get('gran_dur_max', 0.05), step=0.005, key='gran_dur_max')
                gran_amp_min = st.slider("Ampiezza Grani Minima", 0.0, 1.0, st.session_state.get('gran_amp_min', 0.01), step=0.01, key='gran_amp_min')
                gran_amp_max = st.slider("Ampiezza Grani Massima", 0.0, 1.0, st.session_state.get('gran_amp_max', 0.1), step=0.01, key='gran_amp_max')

                params['gran_density_source'] = gran_density_source
                params['gran_duration_source'] = gran_duration_source
                params['gran_amp_source'] = gran_amp_source
                params['gran_density_range'] = (gran_density_min, gran_density_max)
                params['gran_duration_range'] = (gran_duration_min, gran_duration_max)
                params['gran_amp_range'] = (gran_amp_min, gran_amp_max)

                gran_density_data_raw = []
                if gran_density_source == "Dettaglio": gran_density_data_raw = detail_data
                elif gran_density_source == "Movimento": gran_density_data_raw = movement_data
                elif gran_density_source == "Variazione Movimento": gran_density_data_raw = variation_movement_data

                gran_duration_data_raw = []
                if gran_duration_source == "Dettaglio": gran_duration_data_raw = detail_data
                elif gran_duration_source == "Movimento": gran_duration_data_raw = movement_data
                elif gran_duration_source == "Variazione Movimento": gran_duration_data_raw = variation_movement_data

                gran_amp_data_raw = []
                if gran_amp_source == "Dettaglio": gran_amp_data_raw = detail_data
                elif gran_amp_source == "Movimento": gran_amp_data_raw = movement_data
                elif gran_amp_source == "Variazione Movimento": gran_amp_data_raw = variation_movement_data

                gran_density_scaled = np.interp(gran_density_data_raw, (min(gran_density_data_raw) if gran_density_data_raw else 0, max(gran_density_data_raw) if gran_density_data_raw else 1), (gran_density_min, gran_density_max)).tolist()
                gran_duration_scaled = np.interp(gran_duration_data_raw, (min(gran_duration_data_raw) if gran_duration_data_raw else 0, max(gran_duration_data_raw) if gran_duration_data_raw else 1), (gran_duration_min, gran_duration_max)).tolist()
                gran_amp_scaled = np.interp(gran_amp_data_raw, (min(gran_amp_data_raw) if gran_amp_data_raw else 0, max(gran_amp_data_raw) if gran_amp_data_raw else 1), (gran_amp_min, gran_amp_max)).tolist()
            else:
                gran_density_scaled = []
                gran_duration_scaled = []
                gran_amp_scaled = []

        with tab_noise:
            st.markdown("### Layer: Rumore")
            use_noise = st.checkbox("Abilita Rumore", value=st.session_state.get('noise_on', True), key='noise_on')
            params['noise_enabled'] = use_noise
            if use_noise:
                noise_amp_source = st.selectbox("Sorgente Ampiezza Rumore", ["Variazione Movimento", "Movimento", "Dettaglio"], index=["Variazione Movimento", "Movimento", "Dettaglio"].index(st.session_state.get('noise_amp_src', 'Variazione Movimento')), key='noise_amp_src')
                noise_amp_min = st.slider("Ampiezza Minima Rumore", 0.0, 1.0, st.session_state.get('noise_amp_min', 0.0), step=0.01, key='noise_amp_min')
                noise_amp_max = st.slider("Ampiezza Massima Rumore", 0.0, 1.0, st.session_state.get('noise_amp_max', 0.1), step=0.01, key='noise_amp_max')

                params['noise_amp_source'] = noise_amp_source
                params['noise_amp_range'] = (noise_amp_min, noise_amp_max)

                noise_amp_data_raw = []
                if noise_amp_source == "Variazione Movimento": noise_amp_data_raw = variation_movement_data
                elif noise_amp_source == "Movimento": noise_amp_data_raw = movement_data
                elif noise_amp_source == "Dettaglio": noise_amp_data_raw = detail_data

                noise_amp_scaled = np.interp(noise_amp_data_raw, (min(noise_amp_data_raw) if noise_amp_data_raw else 0, max(noise_amp_data_raw) if noise_amp_data_raw else 1), (noise_amp_min, noise_amp_max)).tolist()
            else:
                noise_amp_scaled = []

        with tab_fx:
            st.markdown("### Effetti Audio")
            
            st.subheader("Glitch")
            use_glitch = st.checkbox("Abilita Glitch", value=st.session_state.get('glitch_on', False), key='glitch_on')
            params['glitch_enabled'] = use_glitch
            if use_glitch:
                glitch_factor_source = st.selectbox("Sorgente Fattore Glitch (Probabilità)", ["Variazione Movimento", "Movimento", "Dettaglio"], index=["Variazione Movimento", "Movimento", "Dettaglio"].index(st.session_state.get('glitch_factor_src', 'Variazione Movimento')), key='glitch_factor_src')
                glitch_intensity_source = st.selectbox("Sorgente Intensità Glitch (Durata/Ampiezza)", ["Variazione Movimento", "Movimento", "Dettaglio"], index=["Variazione Movimento", "Movimento", "Dettaglio"].index(st.session_state.get('glitch_intensity_src', 'Variazione Movimento')), key='glitch_intensity_src')
                
                glitch_factor_min = st.slider("Fattore Minimo Glitch (0-1)", 0.0, 1.0, st.session_state.get('glitch_factor_min', 0.01), step=0.005, key='glitch_factor_min')
                glitch_factor_max = st.slider("Fattore Massimo Glitch (0-1)", 0.0, 1.0, st.session_state.get('glitch_factor_max', 0.1), step=0.005, key='glitch_factor_max')
                glitch_intensity_min = st.slider("Intensità Minima Glitch (0-1)", 0.0, 1.0, st.session_state.get('glitch_intensity_min', 0.1), step=0.01, key='glitch_intensity_min')
                glitch_intensity_max = st.slider("Intensità Massima Glitch (0-1)", 0.0, 1.0, st.session_state.get('glitch_intensity_max', 0.8), step=0.01, key='glitch_intensity_max')

                params['glitch_factor_source'] = glitch_factor_source
                params['glitch_intensity_source'] = glitch_intensity_source
                params['glitch_factor_range'] = (glitch_factor_min, glitch_factor_max)
                params['glitch_intensity_range'] = (glitch_intensity_min, glitch_intensity_max)

                glitch_factor_data_raw = []
                if glitch_factor_source == "Variazione Movimento": glitch_factor_data_raw = variation_movement_data
                elif glitch_factor_source == "Movimento": glitch_factor_data_raw = movement_data
                elif glitch_factor_source == "Dettaglio": glitch_factor_data_raw = detail_data
                
                glitch_intensity_data_raw = []
                if glitch_intensity_source == "Variazione Movimento": glitch_intensity_data_raw = variation_movement_data
                elif glitch_intensity_source == "Movimento": glitch_intensity_data_raw = movement_data
                elif glitch_intensity_source == "Dettaglio": glitch_intensity_data_raw = detail_data

                glitch_factor_scaled = np.interp(glitch_factor_data_raw, (min(glitch_factor_data_raw) if glitch_factor_data_raw else 0, max(glitch_factor_data_raw) if glitch_factor_data_raw else 1), (glitch_factor_min, glitch_factor_max)).tolist()
                glitch_intensity_data = np.interp(glitch_intensity_data_raw, (min(glitch_intensity_data_raw) if glitch_intensity_data_raw else 0, max(glitch_intensity_data_raw) if glitch_intensity_data_raw else 1), (glitch_intensity_min, glitch_intensity_max)).tolist()
            else:
                glitch_factor_scaled = []
                glitch_intensity_data = []

            st.subheader("Delay")
            use_delay = st.checkbox("Abilita Delay", value=st.session_state.get('delay_on', False), key='delay_on')
            params['delay_enabled'] = use_delay
            if use_delay:
                delay_time_source = st.selectbox("Sorgente Tempo Delay (sec)", ["Movimento", "Variazione Movimento", "Luminosità"], index=["Movimento", "Variazione Movimento", "Luminosità"].index(st.session_state.get('delay_time_src', 'Movimento')), key='delay_time_src')
                delay_feedback_source = st.selectbox("Sorgente Feedback Delay (0-1)", ["Movimento", "Variazione Movimento", "Dettaglio"], index=["Movimento", "Variazione Movimento", "Dettaglio"].index(st.session_state.get('delay_feedback_src', 'Movimento')), key='delay_feedback_src')
                
                delay_time_min = st.slider("Tempo Minimo Delay (sec)", 0.01, 0.5, st.session_state.get('delay_time_min', 0.1), step=0.01, key='delay_time_min')
                delay_time_max = st.slider("Tempo Massimo Delay (sec)", 0.01, 0.5, st.session_state.get('delay_time_max', 0.3), step=0.01, key='delay_time_max')
                delay_feedback_min = st.slider("Feedback Minimo Delay", 0.0, 0.95, st.session_state.get('delay_feedback_min', 0.3), step=0.01, key='delay_feedback_min')
                delay_feedback_max = st.slider("Feedback Massimo Delay", 0.0, 0.95, st.session_state.get('delay_feedback_max', 0.7), step=0.01, key='delay_feedback_max')

                params['delay_time_source'] = delay_time_source
                params['delay_feedback_source'] = delay_feedback_source
                params['delay_time_range'] = (delay_time_min, delay_time_max)
                params['delay_feedback_range'] = (delay_feedback_min, delay_feedback_max)

                delay_time_data_raw = []
                if delay_time_source == "Movimento": delay_time_data_raw = movement_data
                elif delay_time_source == "Variazione Movimento": delay_time_data_raw = variation_movement_data
                elif delay_time_source == "Luminosità": delay_time_data_raw = luminosity_data

                delay_feedback_data_raw = []
                if delay_feedback_source == "Movimento": delay_feedback_data_raw = movement_data
                elif delay_feedback_source == "Variazione Movimento": delay_feedback_data_raw = variation_movement_data
                elif delay_feedback_source == "Dettaglio": delay_feedback_data_raw = detail_data

                delay_time_scaled = np.interp(delay_time_data_raw, (min(delay_time_data_raw) if delay_time_data_raw else 0, max(delay_time_data_raw) if delay_time_data_raw else 1), (delay_time_min, delay_time_max)).tolist()
                delay_feedback_scaled = np.interp(delay_feedback_data_raw, (min(delay_feedback_data_raw) if delay_feedback_data_raw else 0, max(delay_feedback_data_raw) if delay_feedback_data_raw else 1), (delay_feedback_min, delay_feedback_max)).tolist()
            else:
                delay_time_scaled = []
                delay_feedback_scaled = []


            st.subheader("Riverbero")
            use_reverb = st.checkbox("Abilita Riverbero", value=st.session_state.get('reverb_on', False), key='reverb_on')
            params['reverb_enabled'] = use_reverb
            if use_reverb:
                reverb_decay_source = st.selectbox("Sorgente Tempo Decadimento (sec)", ["Luminosità", "Dettaglio", "Movimento"], index=["Luminosità", "Dettaglio", "Movimento"].index(st.session_state.get('reverb_decay_src', 'Luminosità')), key='reverb_decay_src')
                reverb_mix_source = st.selectbox("Sorgente Mix (Wet/Dry)", ["Luminosità", "Dettaglio", "Movimento"], index=["Luminosità", "Dettaglio", "Movimento"].index(st.session_state.get('reverb_mix_src', 'Luminosità')), key='reverb_mix_src')
                
                reverb_decay_min = st.slider("Decadimento Minimo (sec)", 0.1, 5.0, st.session_state.get('reverb_decay_min', 1.0), step=0.1, key='reverb_decay_min')
                reverb_decay_max = st.slider("Decadimento Massimo (sec)", 0.1, 5.0, st.session_state.get('reverb_decay_max', 3.0), step=0.1, key='reverb_decay_max')
                reverb_mix_min = st.slider("Mix Minimo (0-1)", 0.0, 1.0, st.session_state.get('reverb_mix_min', 0.2), step=0.01, key='reverb_mix_min')
                reverb_mix_max = st.slider("Mix Massimo (0-1)", 0.0, 1.0, st.session_state.get('reverb_mix_max', 0.6), step=0.01, key='reverb_mix_max')

                params['reverb_decay_source'] = reverb_decay_source
                params['reverb_mix_source'] = reverb_mix_source
                params['reverb_decay_range'] = (reverb_decay_min, reverb_decay_max)
                params['reverb_mix_range'] = (reverb_mix_min, reverb_mix_max)

                reverb_decay_data_raw = []
                if reverb_decay_source == "Luminosità": reverb_decay_data_raw = luminosity_data
                elif reverb_decay_source == "Dettaglio": reverb_decay_data_raw = detail_data
                elif reverb_decay_source == "Movimento": reverb_decay_data_raw = movement_data

                reverb_mix_data_raw = []
                if reverb_mix_source == "Luminosità": reverb_mix_data_raw = luminosity_data
                elif reverb_mix_source == "Dettaglio": reverb_mix_data_raw = detail_data
                elif reverb_mix_source == "Movimento": reverb_mix_data_raw = movement_data

                reverb_decay_scaled = np.interp(reverb_decay_data_raw, (min(reverb_decay_data_raw) if reverb_decay_data_raw else 0, max(reverb_decay_data_raw) if reverb_decay_data_raw else 1), (reverb_decay_min, reverb_decay_max)).tolist()
                reverb_mix_scaled = np.interp(reverb_mix_data_raw, (min(reverb_mix_data_raw) if reverb_mix_data_raw else 0, max(reverb_mix_data_raw) if reverb_mix_data_raw else 1), (reverb_mix_min, reverb_mix_max)).tolist()
            else:
                reverb_decay_scaled = []
                reverb_mix_scaled = []


        with tab_eq:
            st.markdown("### Equalizzatore Dinamico")
            use_eq = st.checkbox("Abilita Equalizzatore", value=st.session_state.get('eq_on', False), key='eq_on')
            params['eq_enabled'] = use_eq
            if use_eq:
                eq_low_source = st.selectbox("Sorgente Guadagno Bassi (dB)", ["Luminosità", "Movimento", "Variazione Movimento"], index=["Luminosità", "Movimento", "Variazione Movimento"].index(st.session_state.get('eq_low_src', 'Luminosità')), key='eq_low_src')
                eq_mid_source = st.selectbox("Sorgente Guadagno Medi (dB)", ["Dettaglio", "Luminosità", "Movimento"], index=["Dettaglio", "Luminosità", "Movimento"].index(st.session_state.get('eq_mid_src', 'Dettaglio')), key='eq_mid_src')
                eq_high_source = st.selectbox("Sorgente Guadagno Alti (dB)", ["Movimento", "Dettaglio", "Variazione Movimento"], index=["Movimento", "Dettaglio", "Variazione Movimento"].index(st.session_state.get('eq_high_src', 'Movimento')), key='eq_high_src')
                
                eq_gain_min = st.slider("Guadagno Minimo (dB)", -20.0, 20.0, st.session_state.get('eq_gain_min', -10.0), step=0.5, key='eq_gain_min')
                eq_gain_max = st.slider("Guadagno Massimo (dB)", -20.0, 20.0, st.session_state.get('eq_gain_max', 10.0), step=0.5, key='eq_gain_max')

                params['eq_low_source'] = eq_low_source
                params['eq_mid_source'] = eq_mid_source
                params['eq_high_source'] = eq_high_source
                params['eq_gain_range'] = (eq_gain_min, eq_gain_max)

                eq_low_data_raw = []
                if eq_low_source == "Luminosità": eq_low_data_raw = luminosity_data
                elif eq_low_source == "Movimento": eq_low_data_raw = movement_data
                elif eq_low_source == "Variazione Movimento": eq_low_data_raw = variation_movement_data

                eq_mid_data_raw = []
                if eq_mid_source == "Dettaglio": eq_mid_data_raw = detail_data
                elif eq_mid_source == "Luminosità": eq_mid_data_raw = luminosity_data
                elif eq_mid_source == "Movimento": eq_mid_data_raw = movement_data

                eq_high_data_raw = []
                if eq_high_source == "Movimento": eq_high_data_raw = movement_data
                elif eq_high_source == "Dettaglio": eq_high_data_raw = detail_data
                elif eq_high_source == "Variazione Movimento": eq_high_data_raw = variation_movement_data

                eq_low_scaled = np.interp(eq_low_data_raw, (min(eq_low_data_raw) if eq_low_data_raw else 0, max(eq_low_data_raw) if eq_low_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
                eq_mid_scaled = np.interp(eq_mid_data_raw, (min(eq_mid_data_raw) if eq_mid_data_raw else 0, max(eq_mid_data_raw) if eq_mid_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
                eq_high_scaled = np.interp(eq_high_data_raw, (min(eq_high_data_raw) if eq_high_data_raw else 0, max(eq_high_data_raw) if eq_high_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
            else:
                eq_low_scaled = []
                eq_mid_scaled = []
                eq_high_scaled = []


        st.sidebar.markdown("---")
        # === NUOVO BLOCCO PER SALVARE/CARICARE I PRESET NELLA SIDEBAR ===
        st.sidebar.subheader("Salva e Carica Preset")
        preset_name = st.sidebar.text_input("Nome Preset", value="", key='preset_name')

        if st.sidebar.button("💾 Salva Preset come File"):
            state_to_save = {key: st.session_state[key] for key in st.session_state.keys() if key not in ['uploaded_file']}
            encoded_state = save_state_to_string(state_to_save)
            st.sidebar.download_button(
                label="⬇️ Scarica Preset",
                data=encoded_state,
                file_name=f"{preset_name if preset_name else 'preset'}.json",
                mime="application/json"
            )
            st.sidebar.success("Preset salvato! Clicca sul pulsante 'Scarica Preset'.")

        uploaded_preset_file = st.sidebar.file_uploader("Carica Preset da File", type="json")
        if uploaded_preset_file is not None:
            encoded_state = uploaded_preset_file.getvalue().decode('utf-8')
            loaded_state = load_state_from_string(encoded_state)
            if loaded_state:
                st.session_state.update(loaded_state)
                st.experimental_rerun()
            else:
                st.sidebar.error("❌ Errore nel caricamento del file preset.")

        if st.sidebar.button("🔗 Copia Link con Preset"):
            state_to_save = {key: st.session_state[key] for key in st.session_state.keys() if key not in ['uploaded_file']}
            encoded_state = save_state_to_string(state_to_save)
            query_params_url = {'preset': encoded_state}
            app_url = f"{st.get_app_url()}?{urllib.parse.urlencode(query_params_url)}"
            st.sidebar.text_area("Link Condivisibile", app_url, height=50)
            st.sidebar.success("Link copiato! Puoi incollarlo e condividerlo.")
        # =======================================================


        st.subheader("Impostazioni Output Video")
        output_resolution_choice = st.selectbox("Formato Video Output", list(FORMAT_RESOLUTIONS.keys()), index=list(FORMAT_RESOLUTIONS.keys()).index(st.session_state.get('output_resolution_choice', '16:9 (Orizzontale)')))
        params['output_resolution_choice'] = output_resolution_choice

        col_audio, col_video = st.columns(2)
        with col_audio:
            normalize_audio = st.checkbox("Normalizza Audio Finale", value=st.session_state.get('normalize_audio', True))
            params['normalize_audio'] = normalize_audio
        with col_video:
            use_original_audio = st.checkbox("Mantieni Audio Originale del Video (Mix con quello generato)", value=st.session_state.get('use_original_audio', False))
            params['use_original_audio'] = use_original_audio
            if use_original_audio:
                original_audio_mix_level = st.slider("Livello Mix Audio Originale", 0.0, 1.0, st.session_state.get('original_audio_mix_level', 0.5), step=0.01)
                params['original_audio_mix_level'] = original_audio_mix_level
            else:
                params['original_audio_mix_level'] = 0.0


        if st.button("Genera Video con Audio"):
            st.info("🎵 Generazione e mixaggio audio in corso... Attendere.")
            progress_bar_audio = st.progress(0)
            status_text_audio = st.empty()

            combined_audio = np.zeros(audio_generator.total_samples, dtype=np.float32)

            if use_subtractive:
                subtractive_audio = audio_generator.generate_subtractive_waveform(sub_freq_scaled, sub_amp_scaled, sub_waveform_type)
                combined_audio += subtractive_audio
            
            if use_fm:
                fm_audio = audio_generator.generate_fm_layer(fm_carrier_scaled, fm_mod_scaled, fm_mod_idx_scaled, fm_amp_scaled)
                combined_audio += fm_audio

            if use_granular:
                granular_audio = audio_generator.generate_granular_layer(gran_density_scaled, gran_duration_scaled, gran_amp_scaled)
                combined_audio += granular_audio

            if use_noise:
                noise_audio = audio_generator.add_noise_layer(combined_audio, noise_amp_scaled)
                combined_audio += noise_audio

            progress_bar_audio.progress(30)
            status_text_audio.text("Applicazione effetti audio...")

            if use_glitch:
                combined_audio = audio_generator.apply_glitch_effect(combined_audio, glitch_factor_scaled, glitch_intensity_data)
            
            if use_delay:
                combined_audio = audio_generator.apply_delay_effect(combined_audio, delay_time_scaled, delay_feedback_scaled)

            if use_reverb:
                combined_audio = audio_generator.apply_reverb_effect(combined_audio, reverb_decay_scaled, reverb_mix_scaled)

            if use_eq:
                combined_audio = audio_generator.apply_eq_effect(combined_audio, eq_low_scaled, eq_mid_scaled, eq_high_scaled)

            progress_bar_audio.progress(70)
            status_text_audio.text("Normalizzazione audio...")

            if normalize_audio:
                if np.max(np.abs(combined_audio)) > 1e-6:
                    combined_audio = librosa.util.normalize(combined_audio)
                else:
                    combined_audio = np.zeros_like(combined_audio)
            
            combined_audio = np.clip(combined_audio, -1.0, 1.0)
            
            audio_output_path = "output_audio.wav"
            sf.write(audio_output_path, combined_audio, AUDIO_SAMPLE_RATE)
            
            progress_bar_audio.progress(100)
            status_text_audio.text("Audio generato!")
            st.success("✅ Audio generato con successo!")
            
            gc.collect()

            if not check_ffmpeg():
                st.warning(f"⚠️ FFmpeg non è installato o non è nel PATH. Impossibile unire il video con l'audio. L'audio generato è disponibile in '{audio_output_path}'.")
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV temporaneo)",
                        f,
                        file_name=f"videosound_generato_audio_{base_name_output}.wav",
                        mime="audio/wav"
                    )
                for temp_f in [video_input_path, audio_output_path]:
                    if temp_f and os.path.exists(temp_f):
                        os.remove(temp_f)
                st.info("🗑️ File temporanei puliti.")
            else:
                st.info("🎥 Unione audio/video e ricodifica in corso... Potrebbe richiedere del tempo.")
                progress_bar_video = st.progress(0)
                status_text_video = st.empty()

                final_video_path = f"output_{base_name_output}_{output_resolution_choice.replace(' ', '_')}.mp4"
                
                ffmpeg_command = ["ffmpeg", "-y"]

                if use_original_audio:
                    temp_original_audio_path = "temp_original_audio.aac"
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", video_input_path, "-vn", "-acodec", "aac", temp_original_audio_path
                        ], check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Errore nell'estrazione dell'audio originale: {e.stderr.decode()}")
                        temp_original_audio_path = None

                    if temp_original_audio_path and os.path.exists(temp_original_audio_path):
                        ffmpeg_command.extend([
                            "-i", video_input_path,
                            "-i", audio_output_path,
                            "-i", temp_original_audio_path,
                            "-filter_complex",
                            f"[1:a]volume=1.0[generated_audio];"
                            f"[2:a]volume={original_audio_mix_level}[original_audio];"
                            f"[generated_audio][original_audio]amix=inputs=2:duration=longest[aout]",
                            "-map", "0:v",
                            "-map", "[aout]",
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-crf", "23",
                            "-c:a", "aac",
                            "-b:a", "192k",
                        ])
                    else:
                        st.warning("⚠️ Impossibile estrarre l'audio originale. Verrà usato solo l'audio generato.")
                        ffmpeg_command.extend([
                            "-i", video_input_path,
                            "-i", audio_output_path,
                            "-map", "0:v",
                            "-map", "1:a",
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-crf", "23",
                            "-c:a", "aac",
                            "-b:a", "192k",
                        ])
                else:
                    ffmpeg_command.extend([
                        "-i", video_input_path,
                        "-i", audio_output_path,
                        "-map", "0:v",
                        "-map", "1:a",
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "23",
                        "-c:a", "aac",
                        "-b:a", "192k",
                    ])

                if output_resolution_choice != "Originale":
                    width, height = FORMAT_RESOLUTIONS[output_resolution_choice]
                    ffmpeg_command.extend(["-vf", f"scale={width}:{height},setsar=1:1"])
                
                ffmpeg_command.append(final_video_path)

                try:
                    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    total_seconds = duration_seconds
                    time_pattern = r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})"
                    
                    while True:
                        output = process.stderr.readline()
                        if not output and process.poll() is not None:
                            break
                        if output:
                            time_match = re.search(time_pattern, output)
                            if time_match:
                                hours, minutes, seconds = map(float, time_match.groups())
                                current_seconds = hours * 3600 + minutes * 60 + seconds
                                if total_seconds > 0:
                                    progress = int((current_seconds / total_seconds) * 100)
                                    progress_bar_video.progress(min(progress, 99))
                                    status_text_video.text(f"Elaborazione video: {current_seconds:.2f}/{total_seconds:.2f}s")
                    
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)

                    progress_bar_video.progress(100)
                    status_text_video.text("Video completato!")
                    st.success(f"✅ Video con audio generato con successo! Scarica qui sotto:")
                    
                    with open(final_video_path, "rb") as f:
                        st.download_button(
                            "⬇️ Scarica Video Finale",
                            f,
                            file_name=final_video_path,
                            mime="video/mp4"
                        )
                    
                    for temp_f in [video_input_path, audio_output_path, "temp_original_audio.aac" if use_original_audio else None, final_video_path]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")

                except subprocess.CalledProcessError as e:
                    st.error(f"❌ Errore FFmpeg durante l'unione/ricodifica: {e.stderr.decode()}")
                    st.code(e.stdout.decode() + e.stderr.decode())
                    for temp_f in [video_input_path, audio_output_path, "temp_original_audio.aac" if use_original_audio else None]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")
                except Exception as e:
                    st.error(f"❌ Errore generico durante l'unione/ricodifica: {str(e)}")
                    for temp_f in [video_input_path, audio_output_path, "temp_original_audio.aac" if use_original_audio else None]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")


        st.markdown("---")
        with st.expander("✨ Descrizione del Brano Generato"):
            st.write("Questa è una descrizione dettagliata dei parametri usati per generare il tuo brano:")
            
            st.markdown("#### Impostazioni Video:")
            st.write(f"- Formato Output Video: **{params['output_resolution_choice']}**")
            if params['use_original_audio']:
                st.write(f"- Audio Originale del Video: **Mantenuto** (Livello Mix: **{params['original_audio_mix_level']:.2f}**)")
            else:
                st.write("- Audio Originale del Video: **Non mantenuto**")
            st.write(f"- Normalizzazione Audio Finale: **{'Sì' if params['normalize_audio'] else 'No'}**")
            
            st.markdown("#### Layer Audio:")
            if st.session_state.get('subtractive_on', False):
                st.markdown("##### Sintesi Sottrattiva (Abilitata):")
                st.write(f"- Frequenza Controllata da: **{st.session_state.get('sub_freq_src')}** ({st.session_state.get('sub_freq_min')} - {st.session_state.get('sub_freq_max')} Hz)")
                st.write(f"- Ampiezza Controllata da: **{st.session_state.get('sub_amp_src')}** ({st.session_state.get('sub_amp_min'):.2f} - {st.session_state.get('sub_amp_max'):.2f})")
                st.write(f"- Tipo Onda: **{st.session_state.get('sub_waveform_type')}**")
            else:
                st.write("##### Sintesi Sottrattiva: Disabilitata")

            if st.session_state.get('fm_on', False):
                st.markdown("##### Sintesi FM (Abilitata):")
                st.write(f"- Frequenza Portante Controllata da: **{st.session_state.get('fm_carr_src')}** ({st.session_state.get('fm_carr_min')} - {st.session_state.get('fm_carr_max')} Hz)")
                st.write(f"- Frequenza Modulatore Controllata da: **{st.session_state.get('fm_mod_src')}** ({st.session_state.get('fm_mod_min')} - {st.session_state.get('fm_mod_max')} Hz)")
                st.write(f"- Indice Modulazione Controllato da: **{st.session_state.get('fm_idx_src')}** ({st.session_state.get('fm_idx_min'):.1f} - {st.session_state.get('fm_idx_max'):.1f})")
                st.write(f"- Ampiezza FM Controllata da: **{st.session_state.get('fm_amp_src')}** ({st.session_state.get('fm_amp_min'):.2f} - {st.session_state.get('fm_amp_max'):.2f})")
            else:
                st.write("##### Sintesi FM: Disabilitata")

            if st.session_state.get('granular_on', False):
                st.markdown("##### Sintesi Granulare (Abilitata):")
                st.write(f"- Densità Grani Controllata da: **{st.session_state.get('gran_dens_src')}** ({st.session_state.get('gran_dens_min')} - {st.session_state.get('gran_dens_max')} grani)")
                st.write(f"- Durata Grani Controllata da: **{st.session_state.get('gran_dur_src')}** ({st.session_state.get('gran_dur_min'):.3f} - {st.session_state.get('gran_dur_max'):.3f} sec)")
                st.write(f"- Ampiezza Grani Controllata da: **{st.session_state.get('gran_amp_src')}** ({st.session_state.get('gran_amp_min'):.2f} - {st.session_state.get('gran_amp_max'):.2f})")
            else:
                st.write("##### Sintesi Granulare: Disabilitata")

            if st.session_state.get('noise_on', False):
                st.markdown("##### Rumore (Abilitato):")
                st.write(f"- Ampiezza Rumore Controllata da: **{st.session_state.get('noise_amp_src')}** ({st.session_state.get('noise_amp_min'):.2f} - {st.session_state.get('noise_amp_max'):.2f})")
            else:
                st.write("##### Rumore: Disabilitato")

            st.markdown("#### Effetti Audio:")
            if st.session_state.get('glitch_on', False):
                st.markdown("##### Glitch (Abilitato):")
                st.write(f"- Fattore Glitch (Probabilità) Controllato da: **{st.session_state.get('glitch_factor_src')}** ({st.session_state.get('glitch_factor_min'):.3f} - {st.session_state.get('glitch_factor_max'):.3f})")
                st.write(f"- Intensità Glitch (Durata/Ampiezza) Controllata da: **{st.session_state.get('glitch_intensity_src')}** ({st.session_state.get('glitch_intensity_min'):.2f} - {st.session_state.get('glitch_intensity_max'):.2f})")
            else:
                st.write("##### Glitch: Disabilitato")

            if st.session_state.get('delay_on', False):
                st.markdown("##### Delay (Abilitato):")
                st.write(f"- Tempo Delay Controllato da: **{st.session_state.get('delay_time_src')}** ({st.session_state.get('delay_time_min'):.2f} - {st.session_state.get('delay_time_max'):.2f} sec)")
                st.write(f"- Feedback Delay Controllato da: **{st.session_state.get('delay_feedback_src')}** ({st.session_state.get('delay_feedback_min'):.2f} - {st.session_state.get('delay_feedback_max'):.2f})")
            else:
                st.write("##### Delay: Disabilitato")

            if st.session_state.get('reverb_on', False):
                st.markdown("##### Riverbero (Abilitato):")
                st.write(f"- Tempo Decadimento Controllato da: **{st.session_state.get('reverb_decay_src')}** ({st.session_state.get('reverb_decay_min'):.1f} - {st.session_state.get('reverb_decay_max'):.1f} sec)")
                st.write(f"- Mix Riverbero Controllato da: **{st.session_state.get('reverb_mix_src')}** ({st.session_state.get('reverb_mix_min'):.2f} - {st.session_state.get('reverb_mix_max'):.2f})")
            else:
                st.write("##### Riverbero: Disabilitato")

            if st.session_state.get('eq_on', False):
                st.markdown("##### Equalizzatore Dinamico (Abilitato):")
                st.write(f"- Guadagno Bassi Controllato da: **{st.session_state.get('eq_low_src')}** ({st.session_state.get('eq_gain_min'):.1f} - {st.session_state.get('eq_gain_max'):.1f} dB)")
                st.write(f"- Guadagno Medi Controllato da: **{st.session_state.get('eq_mid_src')}** ({st.session_state.get('eq_gain_min'):.1f} - {st.session_state.get('eq_gain_max'):.1f} dB)")
                st.write(f"- Guadagno Alti Controllato da: **{st.session_state.get('eq_high_src')}** ({st.session_state.get('eq_gain_min'):.1f} - {st.session_state.get('eq_gain_max'):.1f} dB)")
            else:
                st.write("##### Equalizzatore Dinamico: Disabilitato")

    gc.collect()

if __name__ == "__main__":
    main()
