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
        # Correzione: usare len() per verificare se l'array/lista è vuoto
        if len(data_per_frame) == 0:
            return np.zeros(self.total_samples)
        
        # Crea un array di indici temporali per i dati dei frames (es. 0, 1, 2... per ogni frame)
        # Li mappiamo alla durata totale in secondi
        original_time_points = np.linspace(0, self.total_duration_seconds, len(data_per_frame), endpoint=True)
        
        # Interpola i dati dei frame all'array temporale dell'audio
        return np.interp(self.time_array, original_time_points, data_per_frame)

    def generate_subtractive_waveform(self, freq_data: list, amp_data: list, waveform_type: str = "sine") -> np.ndarray:
        """Genera una forma d'onda sottrattiva base con frequenza e ampiezza dinamiche."""
        freq_interp = self._interp_data_to_audio_length(freq_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)
        phase_increment = 2 * np.pi * freq_interp / self.sample_rate
        phase = np.cumsum(phase_increment) # Accumula la fase per la continuità

        if waveform_type == "sine":
            waveform = np.sin(phase)
        elif waveform_type == "square":
            waveform = np.sign(np.sin(phase))
        elif waveform_type == "sawtooth":
            # Per sawtooth, la fase deve essere normalizzata tra 0 e 1 prima della mappatura
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

        # Incrementi di fase per modulatore e portante
        mod_phase_increment = 2 * np.pi * mod_freq_interp / self.sample_rate
        carrier_phase_increment = 2 * np.pi * carrier_freq_interp / self.sample_rate

        # Fasi accumulate
        mod_phase = np.cumsum(mod_phase_increment)
        carrier_phase = np.cumsum(carrier_phase_increment)

        # Modulatore (onda sinusoidale)
        modulator_signal = np.sin(mod_phase)

        # Portante (modulata dal segnale del modulatore)
        # La modulazione aggiunge un offset di fase alla portante, proporzionale all'indice di modulazione e al segnale del modulatore
        carrier_signal = np.sin(carrier_phase + mod_idx_interp * modulator_signal)

        audio = carrier_signal * amp_interp * 0.5 # Attenuato per mix
        return audio

    def generate_granular_layer(self, density_data: list, grain_duration_data: list, amp_data: list) -> np.ndarray:
        """Genera un layer di sintesi granulare."""
        density_interp = self._interp_data_to_audio_length(density_data)
        grain_duration_interp = self._interp_data_to_audio_length(grain_duration_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)
        
        # Il processo granulare è intrinsecamente basato su eventi/tempo.
        # Possiamo definire la "probabilità" di un grano per campione basata sulla densità.
        # Oppure, per maggiore controllo, generare grani a intervalli specifici.

        # Generiamo grani per ogni "frame" logico, ma distribuiti sulla base temporale
        samples_per_virtual_frame = int(self.total_samples / len(density_data)) if len(density_data) > 0 else self.total_samples # Evita divisione per zero
        
        for i in range(len(density_data)):
            current_density = density_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else density_interp[-1]
            current_grain_dur_seconds = grain_duration_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else grain_duration_interp[-1]
            current_amp = amp_interp[i * samples_per_virtual_frame] if i * samples_per_virtual_frame < self.total_samples else amp_interp[-1]

            num_grains_in_segment = int(current_density) # Numero di grani in questo "segmento" temporale
            
            if num_grains_in_segment == 0:
                continue

            grain_dur_samples = int(current_grain_dur_seconds * self.sample_rate)
            grain_dur_samples = max(10, grain_dur_samples) # Minimo 10 campioni per grano

            for _ in range(num_grains_in_segment):
                # Posizione casuale del grano all'interno del segmento corrente
                start_sample_segment = i * samples_per_virtual_frame
                end_sample_segment = min((i + 1) * samples_per_virtual_frame, self.total_samples)

                if start_sample_segment >= end_sample_segment - grain_dur_samples: # Assicurati che ci sia spazio per il grano
                    continue
                
                start_grain_sample = np.random.randint(start_sample_segment, end_sample_segment - grain_dur_samples)

                # Genera un grano (onda sinusoidale casuale) con inviluppo Hanning
                grain_freq = 200 + np.random.rand() * 800 # Frequenza casuale del grano
                grain_t = np.arange(grain_dur_samples) / self.sample_rate
                grain_waveform = np.sin(2 * np.pi * grain_freq * grain_t)

                # Inviluppo Hanning per evitare click
                hanning_window = np.hanning(grain_dur_samples)
                grain_with_envelope = grain_waveform * hanning_window * current_amp * 0.1 # Attenuazione per mix

                # Aggiungi il grano all'audio complessivo
                end_grain_sample = start_grain_sample + grain_dur_samples
                if end_grain_sample <= self.total_samples: # Assicurati che non sfori
                    audio[start_grain_sample:end_grain_sample] += grain_with_envelope
                else: # Se sforasse, tagliamo il grano
                    audio[start_grain_sample:self.total_samples] += grain_with_envelope[:self.total_samples - start_grain_sample]


        return audio

    def add_noise_layer(self, audio_array: np.ndarray, noise_amp_data: list) -> np.ndarray:
        """Aggiunge un layer di rumore modulato all'audio esistente."""
        noise_amp_interp = self._interp_data_to_audio_length(noise_amp_data)
        noise_layer = np.random.normal(0, 1, self.total_samples) * noise_amp_interp * 0.2 # Attenuazione per mix
        return audio_array + noise_layer

    def apply_glitch_effect(self, audio_array: np.ndarray, glitch_factor_data: list, glitch_intensity_data: list) -> np.ndarray:
        """Applica un effetto glitch all'audio."""
        glitched_audio = np.copy(audio_array)
        glitched_audio = np.nan_to_num(glitched_audio, nan=0.0)

        glitch_factor_interp = self._interp_data_to_audio_length(glitch_factor_data)
        glitch_intensity_interp = self._interp_data_to_audio_length(glitch_intensity_data)

        # Itera sui campioni e decidi se applicare un glitch
        # Per evitare glitch per ogni campione, controlliamo a intervalli più ampi (es. ogni 100ms)
        glitch_check_interval_samples = int(0.1 * self.sample_rate) # Controlla ogni 100ms
        
        i = 0
        while i < self.total_samples:
            current_time_idx = min(i, self.total_samples - 1)
            
            if len(glitch_factor_interp) > current_time_idx and np.random.rand() < glitch_factor_interp[current_time_idx]:
                glitch_intensity = glitch_intensity_interp[current_time_idx]
                
                # Durata del glitch basata sull'intensità
                glitch_duration_samples = int(glitch_intensity * self.sample_rate * 0.05) # Max 50ms glitch
                if glitch_duration_samples == 0: glitch_duration_samples = 1 # Min 1 sample
                
                start_glitch_sample = i
                end_glitch_sample = min(start_glitch_sample + glitch_duration_samples, self.total_samples)
                
                if start_glitch_sample < end_glitch_sample:
                    segment = glitched_audio[start_glitch_sample:end_glitch_sample]
                    
                    if len(segment) > 0:
                        glitch_type = np.random.choice(["repeat", "noise", "reverse"])

                        if glitch_type == "repeat":
                            repeat_count = np.random.randint(1, 3)
                            # Assicurati che il tile sia della stessa forma del segmento
                            glitched_segment = np.tile(segment, repeat_count)
                            if glitched_segment.ndim == 1:
                                glitched_segment = glitched_segment[:len(segment)]
                            else: # Stereo
                                glitched_segment = glitched_segment[:segment.shape[0], :]
                            glitched_audio[start_glitch_sample:end_glitch_sample] = glitched_segment
                        elif glitch_type == "noise":
                            # CORREZIONE: Genera il rumore con la stessa forma del segmento
                            glitch_noise = np.random.normal(0, glitch_intensity * 0.5, size=segment.shape)
                            glitched_audio[start_glitch_sample:end_glitch_sample] = glitch_noise
                        elif glitch_type == "reverse":
                            # Per stereo, inverti solo lungo l'asse dei campioni
                            if segment.ndim == 1:
                                glitched_audio[start_glitch_sample:end_glitch_sample] = segment[::-1]
                            else: # Stereo
                                glitched_audio[start_glitch_sample:end_glitch_sample, :] = segment[::-1, :]
                
                i = end_glitch_sample # Salta i campioni glitched
            else:
                i += glitch_check_interval_samples # Passa all'intervallo successivo

        return glitched_audio

    def apply_delay_effect(self, audio_array: np.ndarray, delay_time_data: list, feedback_data: list) -> np.ndarray:
        """Applica un effetto delay dinamico all'audio."""
        # Assicurati che audio_array sia 2D per coerenza se l'audio è stereo
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1) # Converte mono in (N, 1)
        else:
            audio_array_processed = audio_array # Già (N, C)

        delayed_audio = np.copy(audio_array_processed)
        
        delay_time_interp = self._interp_data_to_audio_length(delay_time_data)
        feedback_interp = self._interp_data_to_audio_length(feedback_data)

        num_channels = delayed_audio.shape[1]
        
        # Buffer di delay per ogni canale
        delay_buffers = [np.zeros(self.sample_rate) for _ in range(num_channels)]
        write_indices = [0] * num_channels

        for i in range(len(delayed_audio)):
            current_delay_time_seconds = np.clip(delay_time_interp[i], 0.01, 0.5)
            current_feedback_gain = np.clip(feedback_interp[i], 0.0, 0.95)

            delay_samples = int(current_delay_time_seconds * self.sample_rate)
            
            for c in range(num_channels):
                read_idx = (write_indices[c] - delay_samples + self.sample_rate) % self.sample_rate
                
                current_sample_channel = delayed_audio[i, c]
                delayed_audio[i, c] += delay_buffers[c][read_idx] * current_feedback_gain
                delay_buffers[c][write_indices[c]] = current_sample_channel + delay_buffers[c][read_idx] * current_feedback_gain # Add feedback to buffer

                write_indices[c] = (write_indices[c] + 1) % self.sample_rate

        return delayed_audio.squeeze() if num_channels == 1 else delayed_audio # Ritorna mono se l'input era mono

    def apply_reverb_effect(self, audio_array: np.ndarray, decay_time_data: list, mix_data: list) -> np.ndarray:
        """Applica un semplice effetto di riverbero all'audio."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1) # Converte mono in (N, 1)
        else:
            audio_array_processed = audio_array # Già (N, C)

        reverbed_audio = np.copy(audio_array_processed)
        
        decay_time_interp = self._interp_data_to_audio_length(decay_time_data)
        mix_interp = self._interp_data_to_audio_length(mix_data)

        num_channels = reverbed_audio.shape[1]
        num_delay_lines_per_channel = 4 
        
        # Buffer di delay per ogni linea per ogni canale
        delay_line_buffers = [[np.zeros(self.sample_rate) for _ in range(num_delay_lines_per_channel)] for _ in range(num_channels)]
        write_indices = [[0] * num_delay_lines_per_channel for _ in range(num_channels)]

        fixed_delay_times = np.array([0.029, 0.041, 0.053, 0.067]) # in seconds
        fixed_delay_samples = (fixed_delay_times * self.sample_rate).astype(int)

        for i in range(len(reverbed_audio)):
            current_decay_time_seconds = np.clip(decay_time_interp[i], 0.1, 5.0)
            current_mix_level = np.clip(mix_interp[i], 0.0, 1.0)

            if current_decay_time_seconds > 0:
                feedback_gain_reverb = 10**(-3 / (current_decay_time_seconds * self.sample_rate / np.max(fixed_delay_samples)))
            else:
                feedback_gain_reverb = 0
            feedback_gain_reverb = np.clip(feedback_gain_reverb, 0.0, 0.95)

            for c in range(num_channels):
                wet_signal_channel = 0.0
                for k in range(num_delay_lines_per_channel):
                    read_idx = (write_indices[c][k] - fixed_delay_samples[k] + self.sample_rate) % self.sample_rate
                    
                    wet_signal_channel += delay_line_buffers[c][k][read_idx]
                    
                    delay_line_buffers[c][k][write_indices[c][k]] = (reverbed_audio[i, c] + delay_line_buffers[c][k][read_idx] * feedback_gain_reverb) * 0.5
                    
                    write_indices[c][k] = (write_indices[c][k] + 1) % self.sample_rate
                
                reverbed_audio[i, c] = reverbed_audio[i, c] * (1 - current_mix_level) + wet_signal_channel * current_mix_level
        
        return reverbed_audio.squeeze() if num_channels == 1 else reverbed_audio


    def apply_dynamic_filter(self, audio_array: np.ndarray, cutoff_freq_data: list, resonance_data: list) -> np.ndarray:
        """Applica un filtro passa-basso dinamico con risonanza all'audio stereo."""
        if audio_array.ndim == 1:
            audio_array_stereo = np.stack((audio_array, audio_array), axis=-1)
        else:
            audio_array_stereo = audio_array

        filtered_audio_stereo = np.zeros_like(audio_array_stereo)

        cutoff_freq_interp = self._interp_data_to_audio_length(cutoff_freq_data)
        resonance_interp = self._interp_data_to_audio_length(resonance_data)

        # Inizializza lo stato del filtro per i due canali (per la continuità)
        zi_left = None
        zi_right = None
        
        # Processa in piccoli blocchi per aggiornare i parametri del filtro dinamicamente
        block_size = int(self.sample_rate / 100) # Es. aggiorna ogni 10ms
        if block_size == 0: block_size = 1

        for i in range(0, self.total_samples, block_size):
            current_block_end = min(i + block_size, self.total_samples)
            current_block_len = current_block_end - i

            if current_block_len == 0: continue

            # Prendi i parametri del filtro al punto centrale del blocco
            center_sample_idx = min(i + block_size // 2, self.total_samples - 1)
            
            cutoff_freq = np.clip(cutoff_freq_interp[center_sample_idx], 20, self.sample_rate / 2 - 100)
            Q = np.clip(resonance_interp[center_sample_idx] * 10, 0.5, 20.0)

            if cutoff_freq > 0:
                nyquist = 0.5 * self.sample_rate
                normal_cutoff = cutoff_freq / nyquist
                
                b, a = butter(2, normal_cutoff, btype='low', analog=False)

                segment_left = audio_array_stereo[i:current_block_end, 0]
                segment_right = audio_array_stereo[i:current_block_end, 1]

                if zi_left is None:
                    filtered_left, zi_left = lfilter(b, a, segment_left, zi=np.zeros(max(len(b), len(a)) - 1))
                    filtered_right, zi_right = lfilter(b, a, segment_right, zi=np.zeros(max(len(b), len(a)) - 1))
                else:
                    filtered_left, zi_left = lfilter(b, a, segment_left, zi=zi_left)
                    filtered_right, zi_right = lfilter(b, a, segment_right, zi=zi_right)

                filtered_audio_stereo[i:current_block_end, 0] = filtered_left[:current_block_len]
                filtered_audio_stereo[i:current_block_end, 1] = filtered_right[:current_block_len]
            else:
                filtered_audio_stereo[i:current_block_end] = audio_array_stereo[i:current_block_end]

        return filtered_audio_stereo


    def apply_pitch_time_stretch(self, audio_array: np.ndarray, pitch_shift_data: list, time_stretch_data: list) -> np.ndarray:
        """Applica pitch shift e time stretch dinamici usando librosa."""
        if audio_array.ndim == 2:
            mono_audio = librosa.to_mono(audio_array.T)
        else:
            mono_audio = audio_array

        pitch_shift_interp = self._interp_data_to_audio_length(pitch_shift_data)
        time_stretch_interp = self._interp_data_to_audio_length(time_stretch_data)
        
        # Librosa pitch/stretch lavora meglio su segmenti più lunghi.
        # Useremo un approccio basato su finestre per applicare i parametri dinamici.
        hop_length_stretch = 1024 # Quanto spesso aggiornare i parametri di pitch/stretch
        n_fft_stretch = 4096 # Dimensione della finestra FFT

        stretched_audio = np.zeros(self.total_samples)

        # Dividi l'audio in blocchi per l'elaborazione
        for i in range(0, self.total_samples, hop_length_stretch):
            current_segment_start = i
            current_segment_end = min(i + n_fft_stretch, self.total_samples)
            segment = mono_audio[current_segment_start:current_segment_end]

            if len(segment) == 0: continue

            # Prendi i parametri al centro del segmento
            center_sample_idx = min(current_segment_start + n_fft_stretch // 2, self.total_samples - 1)
            pitch_shift = pitch_shift_interp[center_sample_idx]
            time_stretch_ratio = np.clip(time_stretch_interp[center_sample_idx], 0.5, 2.0)

            # Applica pitch shift
            if pitch_shift != 0:
                segment_pitched = librosa.effects.pitch_shift(
                    y=segment,
                    sr=self.sample_rate,
                    n_steps=pitch_shift,
                    res_type='soxr_hq'
                )
            else:
                segment_pitched = segment

            # Applica time stretch
            if time_stretch_ratio != 1.0 and len(segment_pitched) > 0:
                segment_stretched = librosa.effects.time_stretch(segment_pitched, rate=time_stretch_ratio)
            else:
                segment_stretched = segment_pitched

            # Resample il segmento stretchato per adattarlo alla durata del hop
            target_segment_length = hop_length_stretch
            if i + target_segment_length > self.total_samples:
                target_segment_length = self.total_samples - i

            if len(segment_stretched) > 0 and target_segment_length > 0:
                segment_resampled = librosa.resample(y=segment_stretched, 
                                                     orig_sr=self.sample_rate, 
                                                     target_sr=self.sample_rate, 
                                                     length=target_segment_length)
                
                # Assicurati che segment_resampled abbia la lunghezza esatta richiesta
                if len(segment_resampled) > target_segment_length:
                    segment_resampled = segment_resampled[:target_segment_length]
                elif len(segment_resampled) < target_segment_length:
                    segment_resampled = np.pad(segment_resampled, (0, target_segment_length - len(segment_resampled)))

                stretched_audio[i:i + target_segment_length] = segment_resampled

        # Ri-converti a stereo se necessario (duplicando il canale mono)
        if audio_array.ndim == 2:
            stretched_audio_stereo = np.stack((stretched_audio, stretched_audio), axis=-1)
            return stretched_audio_stereo
        else:
            return stretched_audio

    def apply_modulation_effect(self, audio_array: np.ndarray, modulation_depth_data: list, modulation_rate_data: list) -> np.ndarray:
        """Applica un semplice effetto di vibrato o tremolo di base."""
        modulated_audio = np.copy(audio_array)

        if modulated_audio.ndim == 1:
            num_channels = 1
            modulated_audio = modulated_audio.reshape(-1, 1) # Converte mono in (N, 1)
        else:
            num_channels = modulated_audio.shape[1]

        mod_depth_interp = self._interp_data_to_audio_length(modulation_depth_data)
        mod_rate_interp = self._interp_data_to_audio_length(modulation_rate_data)

        # LFOs per ciascun canale per la continuità
        lfo_phase = [0.0] * num_channels

        # Processa per campione
        for i in range(self.total_samples):
            current_depth = np.clip(mod_depth_interp[i], 0.0, 0.1)
            current_rate = np.clip(mod_rate_interp[i], 0.1, 10.0)

            for c in range(num_channels):
                # Modulazione di ampiezza (Tremolo)
                lfo_value = np.sin(lfo_phase[c])
                modulated_audio[i, c] *= (1 + current_depth * lfo_value)
                lfo_phase[c] += 2 * np.pi * current_rate / self.sample_rate
                lfo_phase[c] %= (2 * np.pi)

        return modulated_audio.squeeze() if num_channels == 1 else modulated_audio

    def apply_panning(self, audio_array: np.ndarray, panning_data: list) -> np.ndarray:
        """Applica il panning dinamico basato sul centro di massa orizzontale.
        Converte l'audio a stereo se mono."""
        if audio_array.ndim == 1:
            stereo_audio = np.stack((audio_array, audio_array), axis=-1)
        else:
            stereo_audio = audio_array

        panned_audio = np.copy(stereo_audio)
        panning_interp = self._interp_data_to_audio_length(panning_data)

        for i in range(self.total_samples):
            pan_value = np.clip(panning_interp[i], 0.0, 1.0)

            gain_left = np.cos(pan_value * np.pi / 2)
            gain_right = np.sin(pan_value * np.pi / 2)

            panned_audio[i, 0] *= gain_left
            panned_audio[i, 1] *= gain_right

        return panned_audio

    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalizza l'audio per evitare clipping."""
        max_abs = np.max(np.abs(audio_array))
        if max_abs > 1.0:
            return audio_array / max_abs
        return audio_array

def analyze_audio_features(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Analizza le caratteristiche di base di un segnale audio.
    
    Args:
        audio_data (np.ndarray): L'array numpy del segnale audio (mono).
        sample_rate (int): La frequenza di campionamento dell'audio.
        
    Returns:
        dict: Un dizionario contenente le caratteristiche analizzate.
    """
    if audio_data.ndim > 1:
        audio_data = librosa.to_mono(audio_data.T)

    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
    avg_rms_db = librosa.amplitude_to_db(np.mean(rms), ref=1.0)

    cent = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=2048, hop_length=512)[0]
    avg_spectral_centroid = np.mean(cent)

    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=2048, hop_length=512)[0]
    avg_zcr = np.mean(zcr)

    return {
        "Durata (secondi)": f"{duration:.2f}",
        "Loudness medio (dB)": f"{avg_rms_db:.2f}",
        "Centroid Spettrale medio (Hz)": f"{avg_spectral_centroid:.2f}",
        "Zero Crossing Rate medio": f"{avg_zcr:.4f}"
    }


def main():
    st.set_page_config(
        page_title="VideoSound Generator",
        page_icon="🎶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎶 VideoSound Generator")
    st.markdown("Crea un paesaggio sonoro dinamico basato sui dati visivi di un video!")

    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg non trovato. Assicurati che FFmpeg sia installato e nel tuo PATH per unire audio e video.")
        st.markdown("[Scarica FFmpeg qui](https://ffmpeg.org/download.html)")

    with st.sidebar:
        st.header("⚙️ Caricamento e Impostazioni")
        uploaded_file = st.file_uploader("Carica un file video (MP4, MOV, AVI, ecc.)", type=["mp4", "mov", "avi", "mkv"])

        st.subheader("Opzioni di Output Video")
        output_resolution_choice = st.selectbox(
            "Seleziona la Risoluzione del Video di Output:",
            list(FORMAT_RESOLUTIONS.keys())
        )

        st.subheader("Opzioni di Download")
        download_audio_only = st.checkbox("Scarica solo Audio (WAV)", value=False)
        
        st.markdown("---")
        st.header("🎛️ Controlli Sintesi Audio")

        enable_subtractive = st.checkbox("Abilita Subtractive Synth", value=True)
        subtractive_waveform = st.selectbox("Forma d'onda Subtractive:", ["sine", "square", "sawtooth"])
        base_freq_subtractive = st.slider("Frequenza Base Subtractive (Hz)", 20, 2000, 440, 10)
        max_freq_multiplier_subtractive = st.slider("Moltiplicatore Freq. Max Subtractive", 1.0, 10.0, 4.0, 0.1)
        amplitude_subtractive = st.slider("Ampiezza Subtractive", 0.0, 1.0, 0.3, 0.01)

        st.markdown("---")
        enable_fm = st.checkbox("Abilita FM Synth", value=False)
        base_carrier_freq_fm = st.slider("Frequenza Portante Base FM (Hz)", 20, 2000, 880, 10, disabled=not enable_fm)
        max_mod_freq_fm = st.slider("Frequenza Modulatore Max FM (Hz)", 10, 1000, 200, 10, disabled=not enable_fm)
        max_mod_idx_fm = st.slider("Indice di Modulazione Max FM", 0.0, 10.0, 5.0, 0.1, disabled=not enable_fm)
        amplitude_fm = st.slider("Ampiezza FM", 0.0, 1.0, 0.2, 0.01, disabled=not enable_fm)

        st.markdown("---")
        enable_granular = st.checkbox("Abilita Granular Synth", value=False)
        base_density_granular = st.slider("Densità Base Grani", 0.0, 20.0, 5.0, 0.1, disabled=not enable_granular)
        min_grain_duration = st.slider("Durata Min Grano (s)", 0.001, 0.1, 0.01, 0.001, disabled=not enable_granular)
        max_grain_duration = st.slider("Durata Max Grano (s)", 0.1, 0.5, 0.2, 0.01, disabled=not enable_granular)
        amplitude_granular = st.slider("Ampiezza Granular", 0.0, 1.0, 0.1, 0.01, disabled=not enable_granular)

        st.markdown("---")
        enable_noise = st.checkbox("Abilita Noise Layer", value=False)
        min_noise_amp = st.slider("Ampiezza Min Rumore", 0.0, 1.0, 0.05, 0.01, disabled=not enable_noise)
        max_noise_amp = st.slider("Ampiezza Max Rumore", 0.0, 1.0, 0.5, 0.01, disabled=not enable_noise)
        
        st.markdown("---")
        st.subheader("Effetti Dinamici")
        enable_dynamic_effects = st.checkbox("Abilita Effetti Dinamici Avanzati", value=False)

        # Inizializza parametri degli effetti anche se i checkbox sono disabilitati
        min_cutoff_adv = 20
        max_cutoff_adv = 20000
        min_resonance_adv = 0.0
        max_resonance_adv = 1.0

        min_glitch_factor = 0.0
        max_glitch_factor = 0.1
        min_glitch_intensity = 0.0
        max_glitch_intensity = 1.0

        min_delay_time = 0.01
        max_delay_time = 0.5
        min_feedback = 0.0
        max_feedback = 0.9

        min_decay_time = 0.1
        max_decay_time = 5.0
        min_reverb_mix = 0.0
        max_reverb_mix = 1.0

        min_pitch_shift = -12
        max_pitch_shift = 12
        min_time_stretch = 0.5
        max_time_stretch = 1.5

        min_mod_depth = 0.0
        max_mod_depth = 0.1
        min_mod_rate = 0.1
        max_mod_rate = 10.0

        with st.expander("F
