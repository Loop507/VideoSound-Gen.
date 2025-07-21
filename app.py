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
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def analyze_video_frames(video_path: str, progress_bar, status_text) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Carica il video, estrae i frame e analizza luminosità, dettaglio/contrasto, movimento e centro di massa orizzontale.
    Restituisce array di luminosità, dettaglio, movimento, variazione di movimento e centro di massa, e info sul video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il file video.")
        return None, None, None, None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("❌ Impossibile leggere il framerate del video.")
        cap.release()
        return None, None, None, None, None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    if video_duration < MIN_DURATION:
        st.error(f"❌ Il video deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {video_duration:.2f}s")
        cap.release()
        return None, None, None, None, None, None, None, None, None
    
    if video_duration > MAX_DURATION:
        st.warning(f"⚠️ Video troppo lungo ({video_duration:.1f}s). Verranno analizzati solo i primi {MAX_DURATION} secondi.")
        total_frames = int(MAX_DURATION * fps)
        video_duration = MAX_DURATION

    brightness_data = []
    detail_data = [] 
    movement_data = [] 
    variation_movement_data = []
    horizontal_center_data = [] # Nuovo array per il centro di massa orizzontale
    
    prev_gray_frame = None 
    prev_movement = 0.0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Luminosità Media: Media dei pixel
        brightness = np.mean(gray_frame) / 255.0 # Normalizza tra 0 e 1
        brightness_data.append(brightness)

        # 2. Dettaglio/Contrasto (usando la deviazione standard o l'energia di Laplace)
        # Un filtro di Laplace è sensibile ai bordi e alle aree di forte gradiente (dettaglio)
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        detail = laplacian.var()
        detail_data.append(detail)

        # 3. Movimento (differenza assoluta tra frame successivi)
        if prev_gray_frame is not None:
            frame_diff = cv2.absdiff(gray_frame, prev_gray_frame)
            movement = np.mean(frame_diff) / 255.0 # Normalizza tra 0 e 1
            movement_data.append(movement)
            
            # Variazione del Movimento
            variation_movement = abs(movement - prev_movement)
            variation_movement_data.append(variation_movement)
            prev_movement = movement
        else:
            movement_data.append(0.0)
            variation_movement_data.append(0.0) # Il primo frame non ha variazione di movimento

        # 4. Centro di Massa Orizzontale
        # Calcola i momenti di immagine
        M = cv2.moments(gray_frame)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
        else:
            cx = width // 2 # Se non ci sono pixel (frame nero), centro
        horizontal_center_data.append(cx / width) # Normalizza tra 0 e 1

        prev_gray_frame = gray_frame
        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"Analisi frame: {i+1}/{total_frames} ({(i+1)/total_frames:.1%})")

    cap.release()
    gc.collect() # Rilascia memoria

    if not brightness_data:
        st.error("❌ Nessun dato estratto dal video. Assicurati che il video non sia corrotto.")
        return None, None, None, None, None, None, None, None, None

    # Normalizza i dati di dettaglio e movimento che non sono già normalizzati
    max_detail = max(detail_data) if detail_data else 1
    if max_detail > 0:
        detail_data = [d / max_detail for d in detail_data]
    
    # Assicurati che movement_data e variation_movement_data abbiano la stessa lunghezza di brightness_data
    # Se il video ha un solo frame, questi saranno vuoti o di lunghezza 0, riempi con 0.0
    if len(movement_data) < len(brightness_data):
        movement_data.extend([0.0] * (len(brightness_data) - len(movement_data)))
    if len(variation_movement_data) < len(brightness_data):
        variation_movement_data.extend([0.0] * (len(brightness_data) - len(variation_movement_data)))

    return np.array(brightness_data), np.array(detail_data), np.array(movement_data), np.array(variation_movement_data), np.array(horizontal_center_data), width, height, fps, video_duration

class AudioGenerator:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.samples_per_frame = self.sample_rate // AUDIO_FPS

    def _apply_filter(self, audio: np.ndarray, cutoff_freq: float, btype: str, order: int = 5) -> np.ndarray:
        """Applica un filtro Butterworth all'audio."""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyquist
        if not 0 < normal_cutoff < 1:
            st.warning(f"⚠️ Frequenza di taglio ({cutoff_freq} Hz) non valida per filtro {btype}. Ignorando il filtro.")
            return audio
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return lfilter(b, a, audio)

    def generate_base_layer(self, total_samples: int, brightness_data: np.ndarray, base_freq: float, brightness_mod_depth: float) -> np.ndarray:
        """Genera lo strato base dell'audio influenzato dalla luminosità."""
        base_audio = np.zeros(total_samples, dtype=np.float32)
        for i, brightness in enumerate(brightness_data):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, total_samples)
            
            if frame_start_sample >= total_samples:
                break

            # La frequenza modula con la luminosità (inverso: più scuro = più basso, più chiaro = più alto)
            # Normalizza la luminosità per farla variare tra 0 e 1
            # Modulazione: 1 - brightness_norm se si vuole che la frequenza diminuisca con la luminosità
            # Se brightness_mod_depth = 1, la frequenza varia tra base_freq/2 e base_freq*1.5 (es.)
            
            # Versione che rende la luminosità 0-1 e la usa per scalare una deviazione dalla base_freq
            # Assumendo brightness_data già normalizzata tra 0 e 1
            mod_factor = (brightness - 0.5) * 2 * brightness_mod_depth # Varia tra -brightness_mod_depth e +brightness_mod_depth
            current_freq = base_freq * (1 + mod_factor)
            
            # Assicurati che la frequenza non sia zero o negativa
            current_freq = max(10.0, current_freq) 

            # Genera una sinusoide per questo frame
            t_frame = np.linspace(0, (frame_end_sample - frame_start_sample) / self.sample_rate, frame_end_sample - frame_start_sample, endpoint=False)
            frame_samples = np.sin(2 * np.pi * current_freq * t_frame)
            
            # Aggiungi all'audio base
            base_audio[frame_start_sample:frame_end_sample] += frame_samples
        return base_audio

    def generate_granular_layer(self, total_samples: int, brightness_data: np.ndarray, movement_data: np.ndarray, detail_data: np.ndarray,
                                min_grain_freq: float, max_grain_freq: float, min_grain_duration: float, max_grain_duration: float,
                                granular_layer_progress, granular_layer_status) -> np.ndarray:
        """Genera lo strato granulare influenzato da luminosità, movimento e dettaglio."""
        granular_audio_layer = np.zeros(total_samples, dtype=np.float32)
        total_frames = len(brightness_data)

        # Assicurati che i dati abbiano la stessa lunghezza
        min_len = min(len(brightness_data), len(movement_data), len(detail_data))
        brightness_data = brightness_data[:min_len]
        movement_data = movement_data[:min_len]
        detail_data = detail_data[:min_len]

        for i in range(min_len):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, total_samples)

            if frame_start_sample >= total_samples:
                break

            # Determina la densità dei grani (più movimento = più grani)
            # Normalizza movement_data tra 0 e 1, poi scala
            # La densità può variare da 1 a un valore massimo, ad esempio 10-20 grani per frame
            movement_norm = movement_data[i] # Assumendo già normalizzato tra 0 e 1
            grain_density = int(1 + movement_norm * 15) # Da 1 a 16 grani per frame

            # Determina la frequenza del grano (più dettaglio = frequenza più alta)
            detail_norm = detail_data[i] # Assumendo già normalizzato
            grain_freq = min_grain_freq + (max_grain_freq - min_grain_freq) * detail_norm

            # Determina la durata del grano (meno luminosità = grani più lunghi)
            # Inverso: 1 - brightness_data[i]
            brightness_inv = 1.0 - brightness_data[i]
            grain_duration_sec = min_grain_duration + (max_grain_duration - min_grain_duration) * brightness_inv
            grain_duration_samples = int(grain_duration_sec * self.sample_rate)

            # Genera grani all'interno del frame corrente
            for _ in range(grain_density):
                if grain_duration_samples == 0:
                    continue

                # Assicurati che la durata del grano non superi la durata del frame
                actual_grain_duration_samples = min(grain_duration_samples, self.samples_per_frame)
                
                # Calcola il massimo offset di partenza possibile per il grano all'interno del frame corrente.
                # Questo assicura che il grano (della lunghezza actual_grain_duration_samples)
                # non si estenda oltre il segmento audio del frame corrente.
                max_start_offset_in_frame = self.samples_per_frame - actual_grain_duration_samples
                
                # Assicurati che l'offset massimo sia non negativo
                max_start_offset_in_frame = max(0, max_start_offset_in_frame)

                # Genera un offset casuale per la partenza del grano all'interno del segmento del frame
                grain_start_offset_within_frame = np.random.randint(0, max_start_offset_in_frame + 1)
                
                # Calcola il punto di partenza assoluto del grano nell'audio totale
                grain_start = frame_start_sample + grain_start_offset_within_frame
                grain_end = grain_start + actual_grain_duration_samples

                # Se il grano esce dalla durata totale dell'audio, tronca
                if grain_end > total_samples:
                    grain_end = total_samples
                    actual_grain_duration_samples = grain_end - grain_start
                    if actual_grain_duration_samples <= 0:
                        continue # Salta questo grano se la sua durata effettiva è zero o negativa

                t_grain = np.linspace(0, actual_grain_duration_samples / self.sample_rate, actual_grain_duration_samples, endpoint=False)
                # Semplice onda sinusoidale per il grano con inviluppo hanning
                grain = np.sin(2 * np.pi * grain_freq * t_grain) * np.hanning(actual_grain_duration_samples)

                granular_audio_layer[grain_start:grain_end] += grain

            granular_layer_progress.progress((i + 1) / total_frames)
            granular_layer_status.text(f"Generazione strato granulare: {i+1}/{total_frames} ({(i+1)/total_frames:.1%})")

        return granular_audio_layer

    def generate_rhythmic_layer(self, total_samples: int, movement_data: np.ndarray, variation_movement_data: np.ndarray,
                                rhythm_freq_min: float, rhythm_freq_max: float) -> np.ndarray:
        """Genera lo strato ritmico influenzato dal movimento e dalla sua variazione."""
        rhythmic_audio_layer = np.zeros(total_samples, dtype=np.float32)
        total_frames = len(movement_data)

        # Assicurati che i dati abbiano la stessa lunghezza
        min_len = min(len(movement_data), len(variation_movement_data))
        movement_data = movement_data[:min_len]
        variation_movement_data = variation_movement_data[:min_len]

        for i in range(min_len):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, total_samples)

            if frame_start_sample >= total_samples:
                break

            # Determina la frequenza del ritmo (più movimento = frequenza più alta)
            movement_norm = movement_data[i] # Assumendo già normalizzato
            rhythm_freq = rhythm_freq_min + (rhythm_freq_max - rhythm_freq_min) * movement_norm

            # Determina l'ampiezza del ritmo (più variazione di movimento = ampiezza più pronunciata)
            variation_movement_norm = variation_movement_data[i] # Assumendo già normalizzato
            amplitude = 0.5 + 0.5 * variation_movement_norm # Varia tra 0.5 e 1.0

            # Genera un'onda quadra (o dente di sega per un suono più harsh) per il ritmo
            t_frame = np.linspace(0, (frame_end_sample - frame_start_sample) / self.sample_rate, frame_end_sample - frame_start_sample, endpoint=False)
            # Onda quadra, puoi sperimentare con np.sin o np.sawtooth
            frame_samples = amplitude * np.sign(np.sin(2 * np.pi * rhythm_freq * t_frame))
            
            rhythmic_audio_layer[frame_start_sample:frame_end_sample] += frame_samples
        return rhythmic_audio_layer

    def generate_spatial_layer(self, total_samples: int, horizontal_center_data: np.ndarray, spatial_depth: float) -> np.ndarray:
        """Genera lo strato spaziale (stereo) influenzato dal centro di massa orizzontale."""
        spatial_audio_layer = np.zeros((total_samples, 2), dtype=np.float32) # Due canali per stereo
        total_frames = len(horizontal_center_data)

        for i in range(total_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, total_samples)
            
            if frame_start_sample >= total_samples:
                break

            # Determina il bilanciamento del pan (da -1 per sinistra a 1 per destra)
            # horizontal_center_data è normalizzato tra 0 e 1.
            # Converti in pan: 0 -> -1 (sinistra), 0.5 -> 0 (centro), 1 -> 1 (destra)
            pan_position = (horizontal_center_data[i] - 0.5) * 2 * spatial_depth # Varia da -spatial_depth a +spatial_depth

            # Calcola l'ampiezza per il canale sinistro e destro
            # Formula di pan semplice: sqrt(0.5 * (1 - pan)) per L, sqrt(0.5 * (1 + pan)) per R
            # Oppure, per controllo più diretto:
            amp_left = np.sqrt(0.5 * (1 - pan_position))
            amp_right = np.sqrt(0.5 * (1 + pan_position))
            
            # Applica il pan a un segnale base (es. rumore bianco o impulso)
            # Per semplicità, usiamo un'onda sinusoidale di riferimento che verrà pansata
            t_frame = np.linspace(0, (frame_end_sample - frame_start_sample) / self.sample_rate, frame_end_sample - frame_start_sample, endpoint=False)
            base_signal = np.sin(2 * np.pi * 440 * t_frame) * 0.1 # Onda a 440Hz, ampiezza piccola

            spatial_audio_layer[frame_start_sample:frame_end_sample, 0] += base_signal * amp_left
            spatial_audio_layer[frame_start_sample:frame_end_sample, 1] += base_signal * amp_right
            
        return spatial_audio_layer

    def generate_audio(self, video_duration: float, brightness_data: np.ndarray, detail_data: np.ndarray,
                       movement_data: np.ndarray, variation_movement_data: np.ndarray, horizontal_center_data: np.ndarray,
                       audio_gen_params: dict,
                       progress_callback=None, status_callback=None) -> np.ndarray:
        """Genera l'audio finale combinando i vari strati."""
        total_samples = int(video_duration * self.sample_rate)
        final_audio = np.zeros(total_samples, dtype=np.float32)

        # 1. Strato Base (Luminosità -> Frequenza)
        status_callback("Generazione strato base...")
        base_layer = self.generate_base_layer(total_samples, brightness_data,
                                              audio_gen_params['base_freq'], audio_gen_params['brightness_mod_depth'])
        final_audio += base_layer
        progress_callback(0.25)
        status_callback("Strato base completato.")

        # 2. Strato Granulare (Movimento, Dettaglio, Luminosità -> Densità, Frequenza, Durata)
        status_callback("Generazione strato granulare...")
        granular_layer = self.generate_granular_layer(total_samples, brightness_data, movement_data, detail_data,
                                                      audio_gen_params['min_grain_freq'], audio_gen_params['max_grain_freq'],
                                                      audio_gen_params['min_grain_duration'], audio_gen_params['max_grain_duration'],
                                                      progress_callback, status_callback)
        final_audio += granular_layer
        progress_callback(0.5)
        status_callback("Strato granulare completato.")

        # 3. Strato Ritmico (Movimento, Variazione Movimento -> Frequenza, Ampiezza)
        status_callback("Generazione strato ritmico...")
        rhythmic_layer = self.generate_rhythmic_layer(total_samples, movement_data, variation_movement_data,
                                                      audio_gen_params['rhythm_freq_min'], audio_gen_params['rhythm_freq_max'])
        final_audio += rhythmic_layer
        progress_callback(0.75)
        status_callback("Strato ritmico completato.")

        # Applica i filtri globali
        if audio_gen_params['lowpass_cutoff'] > 0:
            status_callback("Applicazione filtro passa-basso...")
            final_audio = self._apply_filter(final_audio, audio_gen_params['lowpass_cutoff'], 'low')
        if audio_gen_params['highpass_cutoff'] > 0:
            status_callback("Applicazione filtro passa-alto...")
            final_audio = self._apply_filter(final_audio, audio_gen_params['highpass_cutoff'], 'high')

        # Normalizzazione finale e conversione a stereo
        final_audio /= np.max(np.abs(final_audio)) if np.max(np.abs(final_audio)) > 0 else 1.0
        final_audio_stereo = np.array([final_audio, final_audio]).T # Mono a Stereo

        # Strato Spaziale (Centro Orizzontale -> Pan) - applicato allo strato base selettivamente
        # Questo strato è generato come stereo e poi sommato.
        status_callback("Generazione strato spaziale...")
        spatial_layer_stereo = self.generate_spatial_layer(total_samples, horizontal_center_data, audio_gen_params['spatial_depth'])
        
        # Somma lo strato spaziale all'audio stereo finale
        # Assicurati che le dimensioni siano compatibili
        if final_audio_stereo.shape == spatial_layer_stereo.shape:
             final_audio_stereo += spatial_layer_stereo
        else:
             st.warning("⚠️ Dimensioni dello strato spaziale non corrispondenti. Lo strato spaziale potrebbe non essere applicato correttamente.")


        progress_callback(1.0)
        status_callback("Audio completato!")
        return final_audio_stereo

def generate_audio_description(brightness_data: np.ndarray, detail_data: np.ndarray,
                               movement_data: np.ndarray, variation_movement_data: np.ndarray,
                               horizontal_center_data: np.ndarray) -> str:
    """Genera una descrizione testuale del brano in base ai dati di analisi video."""
    description_parts = []

    # Analisi Luminosità
    avg_brightness = np.mean(brightness_data)
    if avg_brightness < 0.3:
        description_parts.append("Il brano ha un'atmosfera profonda e scura, riflettendo la bassa luminosità del video.")
    elif avg_brightness > 0.7:
        description_parts.append("Caratterizzato da suoni chiari e brillanti, in linea con le immagini luminose del video.")
    else:
        description_parts.append("Presenta un equilibrio sonoro, con frequenze che si adattano alle variazioni di luce.")

    # Analisi Dettaglio/Contrasto
    avg_detail = np.mean(detail_data)
    if avg_detail > 0.6:
        description_parts.append("Le texture sonore sono ricche e complesse, rispecchiando l'alto livello di dettaglio visivo.")
    elif avg_detail < 0.2:
        description_parts.append("Il suono è morbido e sfumato, a causa della scarsità di dettagli nel video.")
    else:
        description_parts.append("Variazioni di dettaglio nel suono si adattano ai contorni visivi del video.")

    # Analisi Movimento
    avg_movement = np.mean(movement_data)
    if avg_movement > 0.5:
        description_parts.append("Dinamiche sonore marcate e ritmi veloci accompagnano il movimento intenso.")
    elif avg_movement < 0.1:
        description_parts.append("Il paesaggio sonoro è calmo e statico, riflettendo la scarsità di movimento.")
    else:
        description_parts.append("Il ritmo del brano segue un flusso moderato, corrispondente al movimento del video.")

    # Analisi Variazione Movimento
    avg_variation_movement = np.mean(variation_movement_data)
    if avg_variation_movement > 0.4:
        description_parts.append("Il brano presenta improvvisi cambiamenti di intensità e pattern ritmici mutevoli.")
    elif avg_variation_movement < 0.1:
        description_parts.append("Una stabilità sonora caratterizza il pezzo, con un ritmo costante.")
    else:
        description_parts.append("Le dinamiche si evolvono gradualmente, riflettendo le transizioni visive.")

    # Analisi Centro di Massa Orizzontale
    avg_horizontal_center = np.mean(horizontal_center_data)
    std_horizontal_center = np.std(horizontal_center_data)

    if std_horizontal_center > 0.2: # Significa che il centro si muove molto
        description_parts.append("La spazialità del suono si sposta e si evolve, seguendo il movimento del fuoco visivo.")
    elif avg_horizontal_center < 0.4:
        description_parts.append("Il suono tende a essere posizionato prevalentemente sul lato sinistro.")
    elif avg_horizontal_center > 0.6:
        description_parts.append("Il suono è concentrato maggiormente sul lato destro.")
    else:
        description_parts.append("Il panorama sonoro è ben bilanciato e centrale.")
    
    final_description = "Descrizione del brano generato:\n\n" + "\n".join([f"- {part}" for part in description_parts])
    return final_description


def main():
    st.set_page_config(page_title="VideoSound Generator", layout="wide", initial_sidebar_state="expanded")
    st.title("🎵 VideoSound Generator")
    st.markdown("Crea colonne sonore uniche dai tuoi video, analizzando luminosità, movimento e dettaglio per modellare l'audio.")

    # Sidebar per i parametri audio
    st.sidebar.header("Parametri Generazione Audio")
    with st.sidebar.expander("Strato Base (Luminosità)", expanded=True):
        base_freq = st.slider("Frequenza Base (Hz)", 20, 1000, 200, 10)
        brightness_mod_depth = st.slider("Profondità Mod. Freq. Luminosità", 0.0, 1.0, 0.5, 0.05)

    with st.sidebar.expander("Strato Granulare (Dettaglio, Movimento, Luminosità)", expanded=False):
        min_grain_freq = st.slider("Freq. Grano Min (Hz)", 100, 5000, 800, 50)
        max_grain_freq = st.slider("Freq. Grano Max (Hz)", 200, 10000, 2000, 50)
        min_grain_duration = st.slider("Durata Min Grano (s)", 0.01, 0.5, 0.05, 0.01)
        max_grain_duration = st.slider("Durata Max Grano (s)", 0.1, 2.0, 0.5, 0.05)
    
    with st.sidebar.expander("Strato Ritmico (Movimento)", expanded=False):
        rhythm_freq_min = st.slider("Freq. Ritmo Min (Hz)", 1, 50, 5, 1)
        rhythm_freq_max = st.slider("Freq. Ritmo Max (Hz)", 10, 200, 30, 5)

    with st.sidebar.expander("Strato Spaziale (Centro Orizzontale)", expanded=False):
        spatial_depth = st.slider("Profondità Spaziale (Pan)", 0.0, 1.0, 0.7, 0.05)

    with st.sidebar.expander("Filtri Globali", expanded=False):
        lowpass_cutoff = st.slider("Filtro Passa-Basso (Hz)", 0, 20000, 18000, 100)
        highpass_cutoff = st.slider("Filtro Passa-Alto (Hz)", 0, 1000, 20, 10)
    
    global_volume = st.sidebar.slider("Volume Generale", 0.0, 2.0, 1.0, 0.1)

    audio_gen_params = {
        'base_freq': base_freq,
        'brightness_mod_depth': brightness_mod_depth,
        'min_grain_freq': min_grain_freq,
        'max_grain_freq': max_grain_freq,
        'min_grain_duration': min_grain_duration,
        'max_grain_duration': max_grain_duration,
        'rhythm_freq_min': rhythm_freq_min,
        'rhythm_freq_max': rhythm_freq_max,
        'spatial_depth': spatial_depth,
        'lowpass_cutoff': lowpass_cutoff,
        'highpass_cutoff': highpass_cutoff,
    }

    # Caricamento video
    st.header("Carica il tuo Video")
    uploaded_file = st.file_uploader("Scegli un file video (MP4, MOV, AVI, ecc.)", type=["mp4", "mov", "avi", "mkv"])

    output_resolution_choice = st.selectbox(
        "Seleziona Risoluzione Output Video",
        list(FORMAT_RESOLUTIONS.keys())
    )

    if uploaded_file is not None:
        if not validate_video_file(uploaded_file):
            return

        st.video(uploaded_file)

        # Controlla FFmpeg
        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non trovato. Assicurati che FFmpeg sia installato e disponibile nel tuo PATH per unire l'audio al video e ricodificarlo.")

        if st.button("🚀 Genera Audio e Unisci al Video"):
            with st.spinner("Analisi del video in corso..."):
                # Salvataggio temporaneo del file caricato
                video_input_path = os.path.join("./temp_video_input" + uploaded_file.name)
                with open(video_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Progress bar e status per l'analisi del video
                analysis_progress = st.progress(0)
                analysis_status = st.empty()

                brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, video_duration = \
                    analyze_video_frames(video_input_path, analysis_progress, analysis_status)

                if brightness_data is None:
                    st.error("❌ Errore durante l'analisi del video o video non valido.")
                    if os.path.exists(video_input_path):
                        os.remove(video_input_path)
                    return

                analysis_status.text("Analisi video completata!")

            with st.spinner("Generazione audio in corso..."):
                audio_gen_progress = st.progress(0)
                audio_gen_status = st.empty()

                audio_generator = AudioGenerator(AUDIO_SAMPLE_RATE)
                final_audio_stereo = audio_generator.generate_audio(
                    video_duration, brightness_data, detail_data, movement_data,
                    variation_movement_data, horizontal_center_data, audio_gen_params,
                    progress_callback=audio_gen_progress.progress,
                    status_callback=audio_gen_status.text
                ) * global_volume
                
                # Pulisci la barra di progresso dell'audio e lo stato
                audio_gen_progress.empty()
                audio_gen_status.empty()
                st.success("🎉 Audio generato con successo!")

                # Genera e mostra la descrizione dell'audio
                st.subheader("Descrizione del Brano Generato")
                audio_description = generate_audio_description(brightness_data, detail_data,
                                                               movement_data, variation_movement_data,
                                                               horizontal_center_data)
                st.markdown(audio_description)


                # Salvataggio audio temporaneo
                base_name_output = os.path.splitext(uploaded_file.name)[0]
                audio_output_path = f"generated_audio_{base_name_output}.wav"
                sf.write(audio_output_path, final_audio_stereo, AUDIO_SAMPLE_RATE)

                st.subheader("Risultato")
                st.audio(audio_output_path, format='audio/wav')

                # Unione video e audio (se FFmpeg è disponibile)
                if check_ffmpeg():
                    final_video_path = f"videosound_generato_{base_name_output}.mp4"
                    
                    target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]

                    # FFmpeg command for re-encoding and merging
                    command = [
                        "ffmpeg",
                        "-y", # Sovrascrivi file di output esistenti
                        "-i", video_input_path,
                        "-i", audio_output_path,
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-shortest"
                    ]

                    if target_width > 0 and target_height > 0:
                        # Calcola il ridimensionamento mantenendo l'aspect ratio e poi croppa o aggiungi padding
                        # Per ora, ridimensionamento semplice mantenendo l'aspect ratio e adattando al contenitore
                        # Puoi aggiungere logica di crop o pad qui se necessario.
                        command.extend([
                            "-vf", f"scale=w='if(gte(iw/ih,{target_width}/{target_height}),{target_width},-2)':h='if(gte(iw/ih,{target_width}/{target_height}),-2,{target_height})',setsar=1:1"
                        ])
                        # Aggiungi crop per centrare se l'aspect ratio non corrisponde esattamente
                        # (es. video 16:9 a 1:1)
                        if output_resolution_choice != "Originale":
                             # Aggiungi crop per adattare alla risoluzione di output
                             command.extend([
                                f"crop={target_width}:{target_height}"
                                f":max(0,(iw-{target_width})/2)"
                                f":max(0,(ih-{target_height})/2)"
                             ])
                    
                    command.append(final_video_path)

                    st.info("🔄 Unione audio e video in corso (potrebbe richiedere tempo)...")
                    try:
                        subprocess.run(command, check=True, capture_output=True, text=True)
                        st.success("✅ Video con audio generato e unito con successo!")
                        
                        st.video(final_video_path)
                        with open(final_video_path, "rb") as f:
                            st.download_button(
                                "⬇️ Scarica Video con Audio",
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


if __name__ == "__main__":
    main()
