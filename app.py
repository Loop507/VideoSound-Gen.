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
AUDIO_FPS = 30 # Frame per secondo dell'audio (dovrebbe corrispondere al video per semplicità)

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

def analyze_video_frames(video_path: str) -> Tuple[list, list, list, list, list, float]:
    """
    Analizza i frame di un video per estrarre dati visivi.

    Args:
        video_path (str): Il percorso del file video da analizzare.

    Returns:
        Tuple[list, list, list, list, list, float]: Una tupla contenente liste di dati
        per luminosità, dettaglio, movimento, variazione del movimento, centro di massa orizzontale,
        e la durata effettiva del video in secondi.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Impossibile aprire il video: {video_path}")
        return [], [], [], [], [], 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = frame_count / fps

    if duration_seconds > MAX_DURATION:
        st.error(f"❌ Video troppo lungo. Durata massima consentita: {MAX_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], 0.0
    if duration_seconds < MIN_DURATION:
        st.error(f"❌ Video troppo corto. Durata minima consentita: {MIN_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], 0.0

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

    return luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data, duration_seconds


class AudioGenerator:
    def __init__(self, sample_rate: int, audio_fps: int):
        self.sample_rate = sample_rate
        self.audio_fps = audio_fps
        self.samples_per_frame = self.sample_rate // self.audio_fps

    def generate_subtractive_waveform(self, duration_frames: int, freq_data: list, amp_data: list, waveform_type: str = "sine") -> np.ndarray:
        """Genera una forma d'onda sottrattiva base con frequenza e ampiezza dinamiche."""
        total_samples = duration_frames * self.samples_per_frame
        audio = np.zeros(total_samples)
        phase = 0.0

        for i in range(duration_frames):
            freq = freq_data[i]
            amp = amp_data[i]

            frame_samples = np.arange(self.samples_per_frame)
            t = frame_samples / self.sample_rate

            if waveform_type == "sine":
                waveform = np.sin(2 * np.pi * freq * t + phase)
            elif waveform_type == "square":
                waveform = np.sign(np.sin(2 * np.pi * freq * t + phase))
            elif waveform_type == "sawtooth":
                waveform = 2 * (t * freq - np.floor(t * freq + 0.5))
            else: # default to sine
                waveform = np.sin(2 * np.pi * freq * t + phase)

            # Applicare l'ampiezza e posizionare nel totale
            start_sample = i * self.samples_per_frame
            end_sample = start_sample + self.samples_per_frame
            audio[start_sample:end_sample] += waveform * amp

            # Aggiorna la fase per la continuità
            phase += 2 * np.pi * freq * (self.samples_per_frame / self.sample_rate)
            phase = phase % (2 * np.pi) # Normalizza fase

        return audio

    def generate_fm_layer(self, duration_frames: int, carrier_freq_data: list, mod_freq_data: list, mod_idx_data: list, amp_data: list) -> np.ndarray:
        """Genera un layer di sintesi FM con parametri dinamici."""
        total_samples = duration_frames * self.samples_per_frame
        audio = np.zeros(total_samples)
        carrier_phase = 0.0
        mod_phase = 0.0

        for i in range(duration_frames):
            Fc = carrier_freq_data[i] # Frequenza portante
            Fm = mod_freq_data[i]     # Frequenza modulante
            mod_idx = mod_idx_data[i] # Indice di modulazione
            amp = amp_data[i]

            frame_samples = np.arange(self.samples_per_frame)
            t = frame_samples / self.sample_rate

            # Modulatore (onda sinusoidale)
            modulator_signal = np.sin(2 * np.pi * Fm * t + mod_phase)

            # Portante (modulata dal segnale del modulatore)
            carrier_signal = np.sin(2 * np.pi * Fc * t + mod_idx * modulator_signal + carrier_phase)

            # Applica l'ampiezza e aggiungi al layer audio
            start_sample = i * self.samples_per_frame
            end_sample = start_sample + self.samples_per_frame
            audio[start_sample:end_sample] += carrier_signal * amp * 0.5 # Attenuato per mix

            # Aggiorna le fasi per la continuità
            carrier_phase += 2 * np.pi * Fc * (self.samples_per_frame / self.sample_rate)
            mod_phase += 2 * np.pi * Fm * (self.samples_per_frame / self.sample_rate)
            carrier_phase = carrier_phase % (2 * np.pi)
            mod_phase = mod_phase % (2 * np.pi)

        return audio

    def generate_granular_layer(self, duration_frames: int, density_data: list, grain_duration_data: list, amp_data: list) -> np.ndarray:
        """Genera un layer di sintesi granulare."""
        total_samples = duration_frames * self.samples_per_frame
        audio = np.zeros(total_samples)

        for i in range(duration_frames):
            density = density_data[i] # Numero di grani per frame
            
            # --- CORREZIONE QUI ---
            # Assicurati che grain_dur_samples sia sempre minore di samples_per_frame
            max_allowed_grain_samples = self.samples_per_frame - 1
            if max_allowed_grain_samples <= 0: # Evita problemi se samples_per_frame è troppo piccolo (non dovrebbe accadere con FPS 30)
                max_allowed_grain_samples = 1 
                
            grain_dur_samples = int(np.clip(grain_duration_data[i] * self.sample_rate, 0.005 * self.sample_rate, max_allowed_grain_samples)) 
            # ---------------------
            
            amp = amp_data[i]

            current_frame_start_sample = i * self.samples_per_frame
            current_frame_end_sample = current_frame_start_sample + self.samples_per_frame

            for _ in range(int(density)):
                # Posizione casuale del grano all'interno del frame
                # L'upper bound del randint è ora garantito > 0
                start_grain_sample = current_frame_start_sample + np.random.randint(0, self.samples_per_frame - grain_dur_samples)

                # Genera un grano (onda sinusoidale casuale) con inviluppo Hanning
                grain_freq = 200 + np.random.rand() * 800 # Frequenza casuale del grano
                grain_t = np.arange(grain_dur_samples) / self.sample_rate
                grain_waveform = np.sin(2 * np.pi * grain_freq * grain_t)

                # Inviluppo Hanning per evitare click
                hanning_window = np.hanning(grain_dur_samples)
                grain_with_envelope = grain_waveform * hanning_window * amp * 0.1 # Attenuazione per mix

                # Aggiungi il grano all'audio complessivo
                end_grain_sample = start_grain_sample + grain_dur_samples
                if end_grain_sample < total_samples:
                    audio[start_grain_sample:end_grain_sample] += grain_with_envelope

        return audio

    def add_noise_layer(self, audio_array: np.ndarray, duration_frames: int, noise_amp_data: list) -> np.ndarray:
        """Aggiunge un layer di rumore modulato all'audio esistente."""
        total_samples = duration_frames * self.samples_per_frame
        noise_layer = np.zeros(total_samples)

        for i in range(duration_frames):
            amp = noise_amp_data[i]
            frame_noise = np.random.normal(0, 1, self.samples_per_frame) * amp * 0.2 # Attenuazione per mix

            start_sample = i * self.samples_per_frame
            end_sample = start_sample + self.samples_per_frame
            noise_layer[start_sample:end_sample] = frame_noise

        return audio_array + noise_layer

    def apply_glitch_effect(self, audio_array: np.ndarray, duration_frames: int, glitch_factor_data: list, glitch_intensity_data: list) -> np.ndarray:
        """Applica un effetto glitch all'audio."""
        glitched_audio = np.copy(audio_array)
        
        # Sostituisci i valori NaN con 0 o un valore sensato
        glitched_audio = np.nan_to_num(glitched_audio, nan=0.0)

        for i in range(duration_frames):
            glitch_factor = glitch_factor_data[i] # Probabilità/frequenza del glitch
            glitch_intensity = glitch_intensity_data[i] # Ampiezza del glitch

            if np.random.rand() < glitch_factor: # Probabilità di attivare un glitch
                start_sample = i * self.samples_per_frame
                glitch_duration_frames = int(glitch_intensity * self.audio_fps * 0.1) # Durata del glitch (es. 0.1 secondi max)
                
                # Assicurati che il glitch non vada oltre la fine dell'audio
                end_sample = min(start_sample + glitch_duration_frames * self.samples_per_frame, len(glitched_audio))
                
                if start_sample < end_sample:
                    segment = glitched_audio[start_sample:end_sample]
                    
                    # Scegli un tipo di glitch casuale
                    glitch_type = np.random.choice(["repeat", "noise", "reverse"])

                    if glitch_type == "repeat" and len(segment) > 0:
                        repeat_count = np.random.randint(1, 3) # Ripeti 1 o 2 volte
                        glitched_segment = np.tile(segment, repeat_count)[:len(segment)] # Taglia alla lunghezza originale
                        glitched_audio[start_sample:end_sample] = glitched_segment
                    elif glitch_type == "noise":
                        glitch_noise = np.random.normal(0, glitch_intensity * 0.5, len(segment))
                        glitched_audio[start_sample:end_sample] = glitch_noise
                    elif glitch_type == "reverse" and len(segment) > 0:
                        glitched_audio[start_sample:end_sample] = segment[::-1]

        return glitched_audio


    def apply_delay_effect(self, audio_array: np.ndarray, duration_frames: int, delay_time_data: list, feedback_data: list) -> np.ndarray:
        """Applica un effetto delay dinamico all'audio."""
        delayed_audio = np.copy(audio_array)
        delay_buffer = np.zeros(self.sample_rate) # Buffer di 1 secondo
        write_idx = 0

        for i in range(len(delayed_audio)):
            # Calcola i parametri per il frame corrente
            frame_idx = i // self.samples_per_frame
            if frame_idx >= duration_frames: # Evita IndexError alla fine
                frame_idx = duration_frames - 1

            delay_time_seconds = np.clip(delay_time_data[frame_idx], 0.01, 0.5) # Ritardo tra 10ms e 500ms
            feedback_gain = np.clip(feedback_data[frame_idx], 0.0, 0.95) # Guadagno del feedback

            # Calcola l'indice di lettura dal buffer
            delay_samples = int(delay_time_seconds * self.sample_rate)
            read_idx = (write_idx - delay_samples + self.sample_rate) % self.sample_rate

            # Applica il delay: uscita = input + feedback * delay_buffer[read_idx]
            # Salva l'output attuale nel buffer per il prossimo feedback
            current_sample = delayed_audio[i]
            delayed_audio[i] += delay_buffer[read_idx] * feedback_gain
            delay_buffer[write_idx] = current_sample + delay_buffer[read_idx] * feedback_gain # Aggiungi feedback al buffer

            write_idx = (write_idx + 1) % self.sample_rate

        return delayed_audio
    
    def apply_reverb_effect(self, audio_array: np.ndarray, duration_frames: int, decay_time_data: list, mix_data: list) -> np.ndarray:
        """Applica un semplice effetto di riverbero all'audio."""
        reverbed_audio = np.copy(audio_array)
        
        # Numero di linee di ritardo per un riverbero semplificato
        num_delay_lines = 4 
        delay_line_buffers = [np.zeros(self.sample_rate) for _ in range(num_delay_lines)] # Buffer di 1 secondo per linea
        write_indices = [0] * num_delay_lines

        # Tempi di ritardo fissi per le linee (dovrebbero essere non correlati)
        fixed_delay_times = [0.029, 0.041, 0.053, 0.067] # in secondi

        for i in range(len(reverbed_audio)):
            frame_idx = i // self.samples_per_frame
            if frame_idx >= duration_frames:
                frame_idx = duration_frames - 1
            
            decay_time_seconds = np.clip(decay_time_data[frame_idx], 0.1, 5.0) # Tempo di decadimento tra 0.1 e 5 secondi
            mix_level = np.clip(mix_data[frame_idx], 0.0, 1.0) # Livello di mix tra dry e wet

            # Calcola il guadagno di feedback per ottenere il tempo di decadimento desiderato
            # feedback_gain = 10**(-3 * fixed_delay_time / decay_time_seconds)
            # Questo è più robusto per il riverbero:
            feedback_gain_reverb = np.power(0.001, (1.0 / (decay_time_seconds * self.sample_rate / max(fixed_delay_times) ))) if decay_time_seconds > 0 else 0 
            feedback_gain_reverb = np.clip(feedback_gain_reverb, 0.0, 0.95)

            wet_signal = 0.0
            for k in range(num_delay_lines):
                delay_samples = int(fixed_delay_times[k] * self.sample_rate)
                read_idx = (write_indices[k] - delay_samples + self.sample_rate) % self.sample_rate
                
                # Aggiungi il campione ritardato al segnale wet
                wet_signal += delay_line_buffers[k][read_idx]
                
                # Aggiorna il buffer della linea di ritardo con il segnale attuale + feedback
                delay_line_buffers[k][write_indices[k]] = (reverbed_audio[i] + delay_line_buffers[k][read_idx] * feedback_gain_reverb) * 0.5 # Attenuazione
                
                write_indices[k] = (write_indices[k] + 1) % self.sample_rate
            
            # Mix tra segnale originale (dry) e segnale riverberato (wet)
            reverbed_audio[i] = reverbed_audio[i] * (1 - mix_level) + wet_signal * mix_level
        
        return reverbed_audio


    def apply_dynamic_filter(self, audio_array: np.ndarray, duration_frames: int, cutoff_freq_data: list, resonance_data: list) -> np.ndarray:
        """Applica un filtro passa-basso dinamico con risonanza all'audio stereo."""
        # Se l'audio è mono, trasformalo in stereo per processare entrambi i canali
        if audio_array.ndim == 1:
            audio_array_stereo = np.stack((audio_array, audio_array), axis=-1)
        else:
            audio_array_stereo = audio_array

        filtered_audio_stereo = np.zeros_like(audio_array_stereo)

        # Inizializza lo stato del filtro per i due canali (per la continuità)
        zi_left = None
        zi_right = None

        for i in range(duration_frames):
            # Ottieni i parametri per il frame corrente
            cutoff_freq = np.clip(cutoff_freq_data[i], 20, self.sample_rate / 2 - 100) # Frequenza di taglio
            Q = np.clip(resonance_data[i] * 10, 0.5, 20.0) # Fattore Q per la risonanza (da 0.5 a 20 circa)

            # Calcola i coefficienti del filtro Butterworth (passa-basso)
            # Ordine 2 per un buon compromesso tra pendenza e prestazioni
            # b, a = butter(2, cutoff_freq, btype='low', fs=self.sample_rate)
            
            # Per risonanza, usiamo un filtro passa-banda con Q elevato oppure un filtro di tipo Peak
            # Semplifichiamo mappando Q sulla larghezza di banda per un bandpass:
            if cutoff_freq > 0: # Evita divisione per zero
                nyquist = 0.5 * self.sample_rate
                normal_cutoff = cutoff_freq / nyquist
                
                # Calcola i coefficienti per un filtro passa-banda.
                # Per una "risonanza" con un passa-basso, in realtà si usa un filtro passa-basso 
                # e si modella il fattore Q che in scipy.signal non è diretto per butter.
                # Per semplicità e per simulare una risonanza che si accentua, possiamo usare
                # un filtro passa-banda stretto che si muove con la frequenza di taglio
                # o un filtro passa-basso con un picco di risonanza.
                # Qui useremo un semplice filtro passa-basso con lfilter, e la risonanza sarà implicita
                # nel mapping del Q o nella combinazione con altri effetti.
                # Se si vuole un vero filtro risonante, serve un biquad filter implementation (più complesso)
                
                # Per un passa-basso:
                b, a = butter(2, normal_cutoff, btype='low', analog=False)

                # Applica il filtro al segmento audio per entrambi i canali
                start_sample = i * self.samples_per_frame
                end_sample = start_sample + self.samples_per_frame

                segment_left = audio_array_stereo[start_sample:end_sample, 0]
                segment_right = audio_array_stereo[start_sample:end_sample, 1]

                if zi_left is None: # Solo alla prima iterazione
                    filtered_left, zi_left = lfilter(b, a, segment_left, zi=np.zeros(max(len(b), len(a)) - 1))
                    filtered_right, zi_right = lfilter(b, a, segment_right, zi=np.zeros(max(len(b), len(a)) - 1))
                else:
                    filtered_left, zi_left = lfilter(b, a, segment_left, zi=zi_left)
                    filtered_right, zi_right = lfilter(b, a, segment_right, zi=zi_right)

                filtered_audio_stereo[start_sample:end_sample, 0] = filtered_left
                filtered_audio_stereo[start_sample:end_sample, 1] = filtered_right
            else: # Se la frequenza di taglio è 0 o negativa, passa il segnale così com'è o mettilo a zero
                start_sample = i * self.samples_per_frame
                end_sample = start_sample + self.samples_per_frame
                filtered_audio_stereo[start_sample:end_sample] = audio_array_stereo[start_sample:end_sample] # O np.zeros(...)

        return filtered_audio_stereo


    def apply_pitch_time_stretch(self, audio_array: np.ndarray, duration_seconds: float, pitch_shift_data: list, time_stretch_data: list) -> np.ndarray:
        """Applica pitch shift e time stretch dinamici usando librosa."""
        # librosa funziona meglio con audio mono
        if audio_array.ndim == 2:
            mono_audio = librosa.to_mono(audio_array.T)
        else:
            mono_audio = audio_array

        stretched_audio = np.array([]) # Inizializza come array vuoto

        # Processa frame per frame per applicare i parametri dinamici
        for i in range(len(pitch_shift_data)):
            pitch_shift = pitch_shift_data[i]
            time_stretch_ratio = np.clip(time_stretch_data[i], 0.5, 2.0) # Ratio tra 0.5 (più lento) e 2.0 (più veloce)

            # Estrai il segmento audio corrente
            start_sample = i * self.samples_per_frame
            end_sample = min((i + 1) * self.samples_per_frame, len(mono_audio))
            segment = mono_audio[start_sample:end_sample]

            if len(segment) > 0:
                # Applica pitch shift
                if pitch_shift != 0:
                    segment_pitched = librosa.effects.pitch_shift(
                        y=segment,
                        sr=self.sample_rate,
                        n_steps=pitch_shift,
                        res_type='soxr_hq' # Alta qualità
                    )
                else:
                    segment_pitched = segment

                # Applica time stretch. Librosa stretch cerca di mantenere il pitch.
                # Per un time stretch frammentato, dobbiamo gestirlo segmenti per segmenti
                # e poi ricampionare o interpolare per la lunghezza desiderata.
                
                # Una time stretch frame per frame è complessa per la continuità.
                # Per ora, applichiamo un time stretch sul segmento e poi riadattiamo la lunghezza.
                if time_stretch_ratio != 1.0:
                    # Stretcha il segmento. Questo cambierà la sua lunghezza.
                    segment_stretched = librosa.effects.time_stretch(segment_pitched, rate=time_stretch_ratio)
                    
                    # Ricampiona per riportare alla lunghezza del frame desiderato
                    target_length = self.samples_per_frame
                    if len(segment_stretched) != target_length:
                        segment_stretched = librosa.resample(segment_stretched, 
                                                              orig_sr=librosa.get_samplerate(y=segment_stretched), # ottieni il sample rate corrente
                                                              target_sr=self.sample_rate, 
                                                              length=target_length)
                else:
                    segment_stretched = segment_pitched
                
                # Concatena i segmenti
                stretched_audio = np.concatenate((stretched_audio, segment_stretched))

        # Assicurati che l'audio finale abbia la durata corretta
        target_total_samples = int(duration_seconds * self.sample_rate)
        if len(stretched_audio) < target_total_samples:
            stretched_audio = np.pad(stretched_audio, (0, target_total_samples - len(stretched_audio)))
        elif len(stretched_audio) > target_total_samples:
            stretched_audio = stretched_audio[:target_total_samples]

        # Ri-converti a stereo se necessario (duplicando il canale mono)
        if audio_array.ndim == 2:
            stretched_audio_stereo = np.stack((stretched_audio, stretched_audio), axis=-1)
            return stretched_audio_stereo
        else:
            return stretched_audio

    def apply_modulation_effect(self, audio_array: np.ndarray, duration_frames: int, modulation_depth_data: list, modulation_rate_data: list) -> np.ndarray:
        """Placeholder per un effetto di modulazione (es. Chorus/Flanger/Phaser).
        Attualmente crea un semplice effetto di vibrato o tremolo di base."""
        modulated_audio = np.copy(audio_array)

        # Se l'audio è stereo, processa entrambi i canali
        if modulated_audio.ndim == 2:
            num_channels = modulated_audio.shape[1]
        else:
            num_channels = 1
            modulated_audio = modulated_audio.reshape(-1, 1) # Trasforma in 2D per coerenza

        for i in range(duration_frames):
            depth = np.clip(modulation_depth_data[i], 0.0, 0.1) # Profondità di modulazione (es. 0-0.1 per ampiezza o frequenza)
            rate = np.clip(modulation_rate_data[i], 0.1, 10.0) # Frequenza della modulazione (Hz)

            # Calcola l'LFO per questo frame
            # Usiamo un LFO sinusoidale per modulare ampiezza o pitch
            lfo_t = np.arange(self.samples_per_frame) / self.sample_rate + (i * self.samples_per_frame) / self.sample_rate
            lfo_signal = np.sin(2 * np.pi * rate * lfo_t)

            # Applica modulazione di ampiezza (tremolo semplice)
            start_sample = i * self.samples_per_frame
            end_sample = start_sample + self.samples_per_frame

            for ch in range(num_channels):
                modulated_audio[start_sample:end_sample, ch] *= (1 + depth * lfo_signal) # Modula ampiezza

        return modulated_audio if num_channels == 1 else modulated_audio.squeeze() if num_channels == 1 else modulated_audio


    def apply_panning(self, audio_array: np.ndarray, duration_frames: int, panning_data: list) -> np.ndarray:
        """Applica il panning dinamico basato sul centro di massa orizzontale.
        Converte l'audio a stereo se mono."""
        # Se l'audio è mono, trasformalo in stereo duplicando il canale
        if audio_array.ndim == 1:
            stereo_audio = np.stack((audio_array, audio_array), axis=-1)
        else:
            stereo_audio = audio_array

        panned_audio = np.copy(stereo_audio)

        for i in range(duration_frames):
            pan_value = np.clip(panning_data[i], 0.0, 1.0) # 0.0 = tutto a sinistra, 1.0 = tutto a destra

            # Calcola i guadagni per i canali sinistro e destro usando una curva a potenza costante
            # Questo assicura che il volume percepito rimanga relativamente costante
            gain_left = np.cos(pan_value * np.pi / 2)
            gain_right = np.sin(pan_value * np.pi / 2)

            start_sample = i * self.samples_per_frame
            end_sample = start_sample + self.samples_per_frame

            panned_audio[start_sample:end_sample, 0] *= gain_left
            panned_audio[start_sample:end_sample, 1] *= gain_right

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
        # Se l'audio è stereo, convertilo in mono per l'analisi
        audio_data = librosa.to_mono(audio_data.T) # Assumi che i canali siano nell'ultima dimensione se 2D

    # Durata
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Loudness (RMS - Root Mean Square)
    # Calcoliamo il loudness su finestre per avere una media più rappresentativa
    rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
    avg_rms_db = librosa.amplitude_to_db(np.mean(rms), ref=1.0) # Converti a dB

    # Spectral Centroid (indica la "brillantezza")
    # Calcolato su finestre
    cent = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=2048, hop_length=512)[0]
    avg_spectral_centroid = np.mean(cent)

    # Zero Crossing Rate (misura la velocità con cui il segnale cambia segno, utile per percussività)
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
        max_grain_duration = st.slider("Durata Max Grano (s)", 0.1, 0.5, 0.2, 0.01, disabled=not enable_granular) # Slider range still allows > samples_per_frame / sample_rate, but np.clip handles it.
        amplitude_granular = st.slider("Ampiezza Granular", 0.0, 1.0, 0.1, 0.01, disabled=not enable_granular)

        st.markdown("---")
        enable_noise = st.checkbox("Abilita Noise Layer", value=False)
        min_noise_amp = st.slider("Ampiezza Min Rumore", 0.0, 1.0, 0.05, 0.01, disabled=not enable_noise)
        max_noise_amp = st.slider("Ampiezza Max Rumore", 0.0, 1.0, 0.5, 0.01, disabled=not enable_noise)
        
        st.markdown("---")
        st.subheader("Effetti Dinamici")
        enable_dynamic_effects = st.checkbox("Abilita Effetti Dinamici Avanzati", value=False)

        # Inizializza parametri degli effetti anche se i checkbox sono disabilitati
        # Verranno usati solo se l'effetto corrispondente è abilitato,
        # ma è buona pratica dar loro valori di default validi.
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

        with st.expander("Filtro Dinamico"):
            enable_filter = st.checkbox("Applica Filtro Dinamico (LPF)", value=False, disabled=not enable_dynamic_effects)
            min_cutoff_adv = st.slider("Min Frequenza Taglio (Hz)", 20, 20000, 100, 10, disabled=not enable_filter)
            max_cutoff_adv = st.slider("Max Frequenza Taglio (Hz)", 20, 20000, 8000, 100, disabled=not enable_filter)
            min_resonance_adv = st.slider("Min Risonanza (Q)", 0.0, 1.0, 0.1, 0.01, disabled=not enable_filter)
            max_resonance_adv = st.slider("Max Risonanza (Q)", 0.0, 1.0, 0.8, 0.01, disabled=not enable_filter)

        with st.expander("Effetto Glitch"):
            enable_glitch = st.checkbox("Applica Glitch Effect", value=False, disabled=not enable_dynamic_effects)
            min_glitch_factor = st.slider("Min Freq. Glitch (0-1)", 0.0, 0.1, 0.001, 0.001, disabled=not enable_glitch)
            max_glitch_factor = st.slider("Max Freq. Glitch (0-1)", 0.0, 0.1, 0.05, 0.001, disabled=not enable_glitch)
            min_glitch_intensity = st.slider("Min Intensità Glitch (0-1)", 0.0, 1.0, 0.1, 0.01, disabled=not enable_glitch)
            max_glitch_intensity = st.slider("Max Intensità Glitch (0-1)", 0.0, 1.0, 0.8, 0.01, disabled=not enable_glitch)

        with st.expander("Effetto Delay"):
            enable_delay = st.checkbox("Applica Delay Effect", value=False, disabled=not enable_dynamic_effects)
            min_delay_time = st.slider("Min Tempo Delay (s)", 0.01, 0.5, 0.05, 0.01, disabled=not enable_delay)
            max_delay_time = st.slider("Max Tempo Delay (s)", 0.01, 0.5, 0.3, 0.01, disabled=not enable_delay)
            min_feedback = st.slider("Min Feedback Delay (0-1)", 0.0, 0.9, 0.2, 0.01, disabled=not enable_delay)
            max_feedback = st.slider("Max Feedback Delay (0-1)", 0.0, 0.9, 0.7, 0.01, disabled=not enable_delay)

        with st.expander("Effetto Riverbero"):
            enable_reverb = st.checkbox("Applica Riverbero", value=False, disabled=not enable_dynamic_effects)
            min_decay_time = st.slider("Min Tempo Decadimento (s)", 0.1, 5.0, 0.5, 0.1, disabled=not enable_reverb)
            max_decay_time = st.slider("Max Tempo Decadimento (s)", 0.1, 5.0, 3.0, 0.1, disabled=not enable_reverb)
            min_reverb_mix = st.slider("Min Mix Riverbero (0-1)", 0.0, 1.0, 0.1, 0.01, disabled=not enable_reverb)
            max_reverb_mix = st.slider("Max Mix Riverbero (0-1)", 0.0, 1.0, 0.6, 0.01, disabled=not enable_reverb)
        
        with st.expander("Pitch Shift & Time Stretch"):
            enable_pitch_stretch = st.checkbox("Applica Pitch Shift & Time Stretch", value=False, disabled=not enable_dynamic_effects)
            min_pitch_shift = st.slider("Min Pitch Shift (semitoni)", -12, 12, -3, 1, disabled=not enable_pitch_stretch)
            max_pitch_shift = st.slider("Max Pitch Shift (semitoni)", -12, 12, 3, 1, disabled=not enable_pitch_stretch)
            min_time_stretch = st.slider("Min Time Stretch Ratio (0.5=lento, 1.5=veloce)", 0.5, 2.0, 0.8, 0.1, disabled=not enable_pitch_stretch)
            max_time_stretch = st.slider("Max Time Stretch Ratio (0.5=lento, 1.5=veloce)", 0.5, 2.0, 1.2, 0.1, disabled=not enable_pitch_stretch)

        with st.expander("Effetto Modulazione (Es. Chorus/Flanger)"):
            enable_modulation = st.checkbox("Applica Effetto Modulazione", value=False, disabled=not enable_dynamic_effects)
            min_mod_depth = st.slider("Min Profondità Modulazione (0-1)", 0.0, 0.1, 0.01, 0.001, disabled=not enable_modulation)
            max_mod_depth = st.slider("Max Profondità Modulazione (0-1)", 0.0, 0.1, 0.05, 0.001, disabled=not enable_modulation)
            min_mod_rate = st.slider("Min Frequenza Modulazione (Hz)", 0.1, 10.0, 0.5, 0.1, disabled=not enable_modulation)
            max_mod_rate = st.slider("Max Frequenza Modulazione (Hz)", 0.1, 10.0, 5.0, 0.1, disabled=not enable_modulation)


    if uploaded_file is not None:
        if validate_video_file(uploaded_file):
            # Salva il file caricato temporaneamente
            video_input_path = os.path.join("/tmp", uploaded_file.name)
            with open(video_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # # Rimosso: st.video per la preview
            # st.video(video_input_path, format="video/mp4", start_time=0) 

            # Analizza i frame del video
            (
                luminosity_data,
                detail_data,
                movement_data,
                variation_movement_data,
                horizontal_mass_center_data,
                duration_seconds
            ) = analyze_video_frames(video_input_path)

            if luminosity_data: # Solo se l'analisi ha prodotto dati validi
                audio_generator = AudioGenerator(AUDIO_SAMPLE_RATE, AUDIO_FPS)
                duration_frames = len(luminosity_data)

                # Generazione dei layer audio
                st.info("🎵 Generazione layer audio in corso...")
                audio_layers = []

                if enable_subtractive:
                    freq_subtractive = np.interp(luminosity_data, [0, 1], [base_freq_subtractive, base_freq_subtractive * max_freq_multiplier_subtractive])
                    audio_layers.append(audio_generator.generate_subtractive_waveform(
                        duration_frames, freq_subtractive, [amplitude_subtractive] * duration_frames, subtractive_waveform
                    ))

                if enable_fm:
                    carrier_freq_fm = np.interp(luminosity_data, [0, 1], [base_carrier_freq_fm, base_carrier_freq_fm * 2])
                    mod_freq_fm = np.interp(movement_data, [0, 1], [10, max_mod_freq_fm])
                    mod_idx_fm = np.interp(detail_data, [0, 1], [0.0, max_mod_idx_fm])
                    audio_layers.append(audio_generator.generate_fm_layer(
                        duration_frames, carrier_freq_fm, mod_freq_fm, mod_idx_fm, [amplitude_fm] * duration_frames
                    ))

                if enable_granular:
                    density_granular = np.interp(movement_data, [0, 1], [base_density_granular, base_density_granular * 5])
                    # Inverso del dettaglio per durata grano: più dettaglio = grani più corti
                    grain_duration_granular = np.interp(detail_data, [0, 1], [max_grain_duration, min_grain_duration])
                    audio_layers.append(audio_generator.generate_granular_layer(
                        duration_frames, density_granular, grain_duration_granular, [amplitude_granular] * duration_frames
                    ))

                combined_audio = np.sum(audio_layers, axis=0) if audio_layers else np.zeros(int(duration_seconds * AUDIO_SAMPLE_RATE))

                if enable_noise:
                    noise_amp = np.interp(detail_data, [0, 1], [min_noise_amp, max_noise_amp])
                    combined_audio = audio_generator.add_noise_layer(combined_audio, duration_frames, noise_amp)

                # Applicazione degli effetti avanzati
                if enable_dynamic_effects:
                    st.info("✨ Applicazione effetti dinamici avanzati...")

                    if enable_filter:
                        # Mappa luminosità alla frequenza di taglio e dettaglio alla risonanza
                        cutoff_freq_data = np.interp(luminosity_data, [0, 1], [min_cutoff_adv, max_cutoff_adv])
                        resonance_data = np.interp(detail_data, [0, 1], [min_resonance_adv, max_resonance_adv])
                        combined_audio = audio_generator.apply_dynamic_filter(combined_audio, duration_frames, cutoff_freq_data, resonance_data)
                    
                    if enable_glitch:
                        # Mappa variazione movimento alla frequenza glitch, movimento all'intensità
                        glitch_factor_data = np.interp(variation_movement_data, [0, np.max(variation_movement_data) if np.max(variation_movement_data) > 0 else 1], [min_glitch_factor, max_glitch_factor])
                        glitch_intensity_data = np.interp(movement_data, [0, 1], [min_glitch_intensity, max_glitch_intensity])
                        combined_audio = audio_generator.apply_glitch_effect(combined_audio, duration_frames, glitch_factor_data, glitch_intensity_data)

                    if enable_delay:
                        # Mappa movimento al tempo di delay (più movimento = delay più corto)
                        delay_time_data = np.interp(movement_data, [0, 1], [max_delay_time, min_delay_time])
                        feedback_data = np.interp(detail_data, [0, 1], [min_feedback, max_feedback])
                        combined_audio = audio_generator.apply_delay_effect(combined_audio, duration_frames, delay_time_data, feedback_data)

                    if enable_reverb:
                        # Mappa luminosità al tempo di decadimento e dettaglio al mix
                        decay_time_data = np.interp(luminosity_data, [0, 1], [min_decay_time, max_decay_time])
                        mix_data = np.interp(detail_data, [0, 1], [min_reverb_mix, max_reverb_mix])
                        combined_audio = audio_generator.apply_reverb_effect(combined_audio, duration_frames, decay_time_data, mix_data)
                    
                    if enable_pitch_stretch:
                        # Mappa luminosità/dettaglio al pitch shift e movimento al time stretch
                        pitch_shift_data = np.interp(luminosity_data, [0, 1], [min_pitch_shift, max_pitch_shift])
                        time_stretch_data = np.interp(movement_data, [0, 1], [min_time_stretch, max_time_stretch])
                        combined_audio = audio_generator.apply_pitch_time_stretch(combined_audio, duration_seconds, pitch_shift_data, time_stretch_data)
                    
                    if enable_modulation:
                        # Mappa dettaglio alla profondità e movimento alla frequenza di modulazione
                        modulation_depth_data = np.interp(detail_data, [0, 1], [min_mod_depth, max_mod_depth])
                        modulation_rate_data = np.interp(movement_data, [0, 1], [min_mod_rate, max_mod_rate])
                        combined_audio = audio_generator.apply_modulation_effect(combined_audio, duration_frames, modulation_depth_data, modulation_rate_data)


                # Applica il panning dinamico
                st.info("🎧 Applicazione panning dinamico...")
                # Assicurati che combined_audio sia 1D prima di passarlo al panning
                if combined_audio.ndim == 2:
                    # Se l'audio è già stereo, il panning lo gestirà.
                    # Se è mono (risultato di librosa), assicuriamoci che sia trattato come tale o convertito
                    if combined_audio.shape[1] == 1: # Ancora 2D ma con 1 canale
                        combined_audio = combined_audio.flatten()
                
                # Se l'audio risultante dagli effetti è ancora mono, assicurati che apply_panning lo renda stereo
                combined_audio_panned = audio_generator.apply_panning(combined_audio, duration_frames, horizontal_mass_center_data)


                # Normalizza l'audio finale
                st.info("🔊 Normalizzazione audio finale...")
                final_audio = audio_generator.normalize_audio(combined_audio_panned)

                # Salva l'audio generato in un file WAV temporaneo
                audio_output_path = os.path.join("/tmp", f"generated_audio_{os.path.basename(video_input_path)}.wav")
                sf.write(audio_output_path, final_audio, AUDIO_SAMPLE_RATE)
                st.success(f"✅ Audio generato e salvato in '{audio_output_path}'")

                base_name_output = os.path.splitext(os.path.basename(uploaded_file.name))[0]

                # --- Inizia la nuova funzionalità di analisi audio ---
                st.subheader("📊 Analisi del Brano Generato")
                audio_analysis_results = analyze_audio_features(final_audio, AUDIO_SAMPLE_RATE)
                analysis_text = "--- Analisi del Brano Generato ---\n\n"
                for key, value in audio_analysis_results.items():
                    analysis_text += f"{key}: {value}\n"
                
                analysis_file_path = os.path.join("/tmp", f"audio_analysis_{base_name_output}.txt")
                with open(analysis_file_path, "w") as f:
                    f.write(analysis_text)
                
                st.text("Ecco un riepilogo delle caratteristiche audio:")
                st.code(analysis_text)
                st.download_button(
                    "⬇️ Scarica Analisi Audio (TXT)",
                    data=analysis_text.encode('utf-8'),
                    file_name=f"audio_analysis_{base_name_output}.txt",
                    mime="text/plain"
                )
                # --- Fine della nuova funzionalità di analisi audio ---


                if download_audio_only:
                    with open(audio_output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Scarica Solo Audio (WAV)",
                            f,
                            file_name=f"videosound_generato_audio_{base_name_output}.wav",
                            mime="audio/wav"
                        )
                else:
                    if ffmpeg_available:
                        # Ricodifica il video e unisci l'audio
                        st.info("🎥 Unione video e audio con FFmpeg in corso...")
                        final_video_path = os.path.join("/tmp", f"final_video_with_audio_{base_name_output}.mp4")

                        # Ottieni la risoluzione desiderata
                        target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]

                        # Costruisci i comandi FFmpeg
                        ffmpeg_command = [
                            "ffmpeg",
                            "-i", video_input_path,
                            "-i", audio_output_path,
                            "-c:v", "libx264",
                            "-preset", "fast",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-map", "0:v:0", # Mappa il primo stream video dall'input 0
                            "-map", "1:a:0", # Mappa il primo stream audio dall'input 1
                        ]
                        
                        if output_resolution_choice != "Originale":
                            ffmpeg_command.extend(["-vf", f"scale={target_width}:{target_height},setsar=1:1"]) # imposta il pixel aspect ratio a 1:1
                        
                        ffmpeg_command.append(final_video_path)

                        try:
                            # Esegui il comando FFmpeg
                            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                            
                            st.success("✅ Video con audio generato con successo!")
                            with open(final_video_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Scarica Video Finale (MP4)",
                                    f,
                                    file_name=f"videosound_generato_{base_name_output}_{output_resolution_choice.replace(' ', '_')}.mp4",
                                    mime="video/mp4"
                                )
                        except subprocess.CalledProcessError as e:
                            # CORREZIONE: Rimosso .decode() perché l'output è già una stringa con text=True
                            st.error(f"❌ Errore FFmpeg durante l'unione/ricodifica: {e.stderr}")
                            st.code(e.stdout + e.stderr) 
                        except Exception as e:
                            st.error(f"❌ Errore generico durante l'unione/ricodifica: {str(e)}")
                        finally: # Questo blocco viene sempre eseguito
                            # Pulizia:
                            for temp_f in [video_input_path, audio_output_path, final_video_path, analysis_file_path]:
                                if os.path.exists(temp_f):
                                    os.remove(temp_f)
                            st.info("🗑️ File temporanei puliti.")

                    else:
                        st.warning(f"⚠️ FFmpeg non trovato. Il video con audio non può essere unito o ricodificato. L'audio generato è disponibile in '{audio_output_path}'.")
                        with open(audio_output_path, "rb") as f:
                            st.download_button(
                                "⬇️ Scarica Solo Audio (WAV temporaneo)",
                                f,
                                file_name=f"videosound_generato_audio_{base_name_output}.wav",
                                mime="audio/wav"
                            )


if __name__ == "__main__":
    main()
