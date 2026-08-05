import streamlit as st
import numpy as np
import cv2
import os
import glob
import subprocess
import gc
import shutil
import zipfile
import hashlib
import uuid
from typing import Tuple
import soundfile as sf
from scipy.signal import butter, lfilter
import re # Spostato qui per assicurare che sia importato all'inizio del modulo

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

# ── PRESET PER TIPO DI VIDEO ────────────────────────────────────────────────
# Ogni preset imposta direttamente i valori di session_state associati alle
# key dei widget. Devono essere scritti PRIMA che i widget vengano istanziati,
# altrimenti Streamlit ignora il valore preimpostato.
PRESET_MANUALE = "🎛️ Manuale (nessun preset)"

VIDEO_PRESETS = {
    "🌿 Natura / Paesaggio": {
        'subtractive_on': True, 'sub_freq_src': "Luminosità", 'sub_amp_src': "Luminosità",
        'sub_waveform_type': "sine", 'sub_freq_min': 80, 'sub_freq_max': 400,
        'sub_amp_min': 0.15, 'sub_amp_max': 0.45,
        'fm_on': False, 'granular_on': False, 'noise_on': False, 'glitch_on': False, 'delay_on': False,
        'reverb_on': True, 'reverb_decay_src': "Luminosità", 'reverb_mix_src': "Luminosità",
        'reverb_decay_min': 2.0, 'reverb_decay_max': 4.5, 'reverb_mix_min': 0.3, 'reverb_mix_max': 0.6,
        'eq_on': False,
    },
    "🏙️ Città / Folla / Traffico": {
        'subtractive_on': False,
        'fm_on': True, 'fm_carr_src': "Movimento", 'fm_mod_src': "Variazione Movimento",
        'fm_idx_src': "Variazione Movimento", 'fm_amp_src': "Movimento",
        'fm_carr_min': 300, 'fm_carr_max': 1200, 'fm_mod_min': 80, 'fm_mod_max': 300,
        'fm_idx_min': 1.0, 'fm_idx_max': 6.0, 'fm_amp_min': 0.1, 'fm_amp_max': 0.35,
        'granular_on': True, 'gran_dens_src': "Dettaglio", 'gran_dur_src': "Movimento", 'gran_amp_src': "Movimento",
        'gran_dens_min': 3, 'gran_dens_max': 9, 'gran_dur_min': 0.015, 'gran_dur_max': 0.04,
        'gran_amp_min': 0.03, 'gran_amp_max': 0.15,
        'noise_on': True, 'noise_amp_src': "Variazione Movimento", 'noise_amp_min': 0.02, 'noise_amp_max': 0.15,
        'glitch_on': False,
        'delay_on': True, 'delay_time_src': "Movimento", 'delay_feedback_src': "Movimento",
        'delay_time_min': 0.03, 'delay_time_max': 0.12, 'delay_feedback_min': 0.2, 'delay_feedback_max': 0.5,
        'reverb_on': False, 'eq_on': False,
    },
    "💃 Corpo in Movimento / Danza": {
        'subtractive_on': True, 'sub_freq_src': "Luminosità", 'sub_amp_src': "Movimento",
        'sub_waveform_type': "sine", 'sub_freq_min': 100, 'sub_freq_max': 500,
        'sub_amp_min': 0.1, 'sub_amp_max': 0.4,
        'fm_on': True, 'fm_carr_src': "Movimento", 'fm_mod_src': "Movimento",
        'fm_idx_src': "Movimento", 'fm_amp_src': "Movimento",
        'fm_carr_min': 200, 'fm_carr_max': 1000, 'fm_mod_min': 40, 'fm_mod_max': 200,
        'fm_idx_min': 0.5, 'fm_idx_max': 4.0, 'fm_amp_min': 0.1, 'fm_amp_max': 0.35,
        'granular_on': False, 'noise_on': False, 'glitch_on': False, 'delay_on': False,
        'reverb_on': False,
        'eq_on': True, 'eq_low_src': "Movimento", 'eq_mid_src': "Movimento", 'eq_high_src': "Movimento",
        'eq_gain_min': -8.0, 'eq_gain_max': 8.0,
    },
    "📼 Found Footage / VHS": {
        'subtractive_on': False,
        'fm_on': True, 'fm_carr_src': "Variazione Movimento", 'fm_mod_src': "Variazione Movimento",
        'fm_idx_src': "Dettaglio", 'fm_amp_src': "Variazione Movimento",
        'fm_carr_min': 150, 'fm_carr_max': 900, 'fm_mod_min': 30, 'fm_mod_max': 200,
        'fm_idx_min': 2.0, 'fm_idx_max': 8.0, 'fm_amp_min': 0.1, 'fm_amp_max': 0.3,
        'granular_on': True, 'gran_dens_src': "Variazione Movimento", 'gran_dur_src': "Dettaglio",
        'gran_amp_src': "Variazione Movimento",
        'gran_dens_min': 2, 'gran_dens_max': 8, 'gran_dur_min': 0.02, 'gran_dur_max': 0.06,
        'gran_amp_min': 0.05, 'gran_amp_max': 0.2,
        'noise_on': True, 'noise_amp_src': "Variazione Movimento", 'noise_amp_min': 0.1, 'noise_amp_max': 0.4,
        'glitch_on': True, 'glitch_character': "Sporco / Analogico (rumore dominante)", 'glitch_factor_src': "Variazione Movimento", 'glitch_intensity_src': "Variazione Movimento",
        'glitch_factor_min': 0.1, 'glitch_factor_max': 0.4, 'glitch_intensity_min': 0.3, 'glitch_intensity_max': 0.8,
        'delay_on': False, 'reverb_on': False, 'eq_on': False,
    },
    "🕰️ Timelapse / Cambi Luce Lenti": {
        'subtractive_on': True, 'sub_freq_src': "Luminosità", 'sub_amp_src': "Dettaglio",
        'sub_waveform_type': "sine", 'sub_freq_min': 60, 'sub_freq_max': 350,
        'sub_amp_min': 0.1, 'sub_amp_max': 0.4,
        'fm_on': True, 'fm_carr_src': "Dettaglio", 'fm_mod_src': "Luminosità",
        'fm_idx_src': "Luminosità", 'fm_amp_src': "Dettaglio",
        'fm_carr_min': 100, 'fm_carr_max': 600, 'fm_mod_min': 20, 'fm_mod_max': 120,
        'fm_idx_min': 0.2, 'fm_idx_max': 3.0, 'fm_amp_min': 0.05, 'fm_amp_max': 0.2,
        'granular_on': False, 'noise_on': False, 'glitch_on': False, 'delay_on': False,
        'reverb_on': True, 'reverb_decay_src': "Luminosità", 'reverb_mix_src': "Luminosità",
        'reverb_decay_min': 2.5, 'reverb_decay_max': 5.0, 'reverb_mix_min': 0.4, 'reverb_mix_max': 0.7,
        'eq_on': False,
    },
    "◼️ Op Art / Pattern Geometrici": {
        'subtractive_on': True, 'sub_freq_src': "Dettaglio", 'sub_amp_src': "Dettaglio",
        'sub_waveform_type': "square", 'sub_freq_min': 150, 'sub_freq_max': 900,
        'sub_amp_min': 0.15, 'sub_amp_max': 0.4,
        'fm_on': True, 'fm_carr_src': "Luminosità", 'fm_mod_src': "Dettaglio",
        'fm_idx_src': "Dettaglio", 'fm_amp_src': "Luminosità",
        'fm_carr_min': 300, 'fm_carr_max': 1500, 'fm_mod_min': 50, 'fm_mod_max': 250,
        'fm_idx_min': 1.0, 'fm_idx_max': 5.0, 'fm_amp_min': 0.05, 'fm_amp_max': 0.2,
        'granular_on': False, 'noise_on': False,
        'glitch_on': True, 'glitch_character': "Pulito / Digitale (repeat + reverse, poco rumore)", 'glitch_factor_src': "Dettaglio", 'glitch_intensity_src': "Dettaglio",
        'glitch_factor_min': 0.02, 'glitch_factor_max': 0.12, 'glitch_intensity_min': 0.1, 'glitch_intensity_max': 0.4,
        'delay_on': False, 'reverb_on': False,
        'eq_on': True, 'eq_low_src': "Luminosità", 'eq_mid_src': "Dettaglio", 'eq_high_src': "Dettaglio",
        'eq_gain_min': -5.0, 'eq_gain_max': 12.0,
    },
    "🌀 Video Sperimentale / Astratto": {
        'subtractive_on': False,
        'fm_on': True, 'fm_carr_src': "Variazione Movimento", 'fm_mod_src': "Variazione Movimento",
        'fm_idx_src': "Variazione Movimento", 'fm_amp_src': "Variazione Movimento",
        'fm_carr_min': 200, 'fm_carr_max': 1800, 'fm_mod_min': 60, 'fm_mod_max': 450,
        'fm_idx_min': 3.0, 'fm_idx_max': 9.0, 'fm_amp_min': 0.1, 'fm_amp_max': 0.3,
        'granular_on': True, 'gran_dens_src': "Dettaglio", 'gran_dur_src': "Variazione Movimento",
        'gran_amp_src': "Dettaglio",
        'gran_dens_min': 2, 'gran_dens_max': 7, 'gran_dur_min': 0.01, 'gran_dur_max': 0.05,
        'gran_amp_min': 0.03, 'gran_amp_max': 0.15,
        'noise_on': False,
        'glitch_on': True, 'glitch_character': "Bilanciato (default)", 'glitch_factor_src': "Variazione Movimento", 'glitch_intensity_src': "Variazione Movimento",
        'glitch_factor_min': 0.05, 'glitch_factor_max': 0.2, 'glitch_intensity_min': 0.2, 'glitch_intensity_max': 0.6,
        'delay_on': True, 'delay_time_src': "Variazione Movimento", 'delay_feedback_src': "Variazione Movimento",
        'delay_time_min': 0.05, 'delay_time_max': 0.25, 'delay_feedback_min': 0.3, 'delay_feedback_max': 0.6,
        'reverb_on': False, 'eq_on': False,
    },
    "🎞️ Audiovisual Sequences": {
        'subtractive_on': True, 'sub_freq_src': "Movimento", 'sub_amp_src': "Movimento",
        'sub_waveform_type': "sine", 'sub_freq_min': 100, 'sub_freq_max': 700,
        'sub_amp_min': 0.15, 'sub_amp_max': 0.45,
        'fm_on': False, 'granular_on': False, 'noise_on': False,
        'glitch_on': True, 'glitch_character': "Pulito / Digitale (repeat + reverse, poco rumore)", 'glitch_factor_src': "Movimento", 'glitch_intensity_src': "Movimento",
        'glitch_factor_min': 0.01, 'glitch_factor_max': 0.08, 'glitch_intensity_min': 0.1, 'glitch_intensity_max': 0.35,
        'delay_on': True, 'delay_time_src': "Movimento", 'delay_feedback_src': "Movimento",
        'delay_time_min': 0.05, 'delay_time_max': 0.15, 'delay_feedback_min': 0.2, 'delay_feedback_max': 0.4,
        'reverb_on': False, 'eq_on': False,
    },
    "⚡ Glitch Digitale / Datamosh": {
        'subtractive_on': False, 'fm_on': False,
        'granular_on': True, 'gran_dens_src': "Variazione Movimento", 'gran_dur_src': "Variazione Movimento",
        'gran_amp_src': "Dettaglio",
        'gran_dens_min': 4, 'gran_dens_max': 10, 'gran_dur_min': 0.01, 'gran_dur_max': 0.025,
        'gran_amp_min': 0.05, 'gran_amp_max': 0.2,
        'noise_on': True, 'noise_amp_src': "Dettaglio", 'noise_amp_min': 0.0, 'noise_amp_max': 0.05,
        'glitch_on': True, 'glitch_character': "Pulito / Digitale (repeat + reverse, poco rumore)", 'glitch_factor_src': "Variazione Movimento", 'glitch_intensity_src': "Variazione Movimento",
        'glitch_factor_min': 0.15, 'glitch_factor_max': 0.5, 'glitch_intensity_min': 0.4, 'glitch_intensity_max': 0.9,
        'delay_on': False, 'reverb_on': False, 'eq_on': False,
    },
    "🛸 Suoni dello Spazio (Barron)": {
        # Ispirato alle "electronic tonalities" di Bebe e Louis Barron per Forbidden Planet (1956):
        # non i momenti "d'allarme" più aggressivi della colonna sonora, ma il carattere più iconico
        # e ricorrente — droni sospesi, quasi immobili, che sembrano respirare nel vuoto siderale.
        # La Sottrattiva è il drone portante, l'FM colora appena (non domina più con ring-mod dura).
        'subtractive_on': True, 'sub_freq_src': "Luminosità", 'sub_amp_src': "Movimento",
        'sub_waveform_type': "sine", 'sub_freq_min': 60, 'sub_freq_max': 800,
        'sub_amp_min': 0.15, 'sub_amp_max': 0.4, 'sub_gain': 1.0,
        'fm_on': True, 'fm_carr_src': "Movimento", 'fm_mod_src': "Variazione Movimento",
        'fm_idx_src': "Variazione Movimento", 'fm_amp_src': "Luminosità",
        'fm_carr_min': 150, 'fm_carr_max': 900, 'fm_mod_min': 20, 'fm_mod_max': 150,
        'fm_idx_min': 0.5, 'fm_idx_max': 3.0, 'fm_amp_min': 0.05, 'fm_amp_max': 0.15, 'fm_gain': 0.6,
        'granular_on': True, 'gran_dens_src': "Dettaglio", 'gran_dur_src': "Movimento",
        'gran_amp_src': "Movimento", 'gran_pitch_src': "Luminosità",
        'gran_dens_min': 1, 'gran_dens_max': 3, 'gran_dur_min': 0.05, 'gran_dur_max': 0.1,
        'gran_amp_min': 0.02, 'gran_amp_max': 0.08, 'gran_pitch_min': 200, 'gran_pitch_max': 1200, 'gran_gain': 0.4,
        'noise_on': False,
        'glitch_on': False,
        'delay_on': True, 'delay_time_src': "Movimento", 'delay_feedback_src': "Variazione Movimento",
        'delay_time_min': 0.1, 'delay_time_max': 0.3, 'delay_feedback_min': 0.2, 'delay_feedback_max': 0.45,
        'reverb_on': True, 'reverb_decay_src': "Luminosità", 'reverb_mix_src': "Luminosità",
        'reverb_decay_min': 3.5, 'reverb_decay_max': 5.0, 'reverb_mix_min': 0.5, 'reverb_mix_max': 0.8,
        'eq_on': True, 'eq_low_src': "Movimento", 'eq_mid_src': "Dettaglio", 'eq_high_src': "Luminosità",
        'eq_gain_min': -5.0, 'eq_gain_max': 6.0,
        'panning_on': True, 'pan_src': "Centro di Massa Orizzontale",
    },
    "📻 Musica Concreta / Nastro": {
        # Ispirato a Schaeffer/Stockhausen e all'estetica dei registratori a nastro: materiale
        # "trovato" e manipolato invece di toni puri, tagli/splice netti, varispeed, fruscio,
        # eco da tape-delay, gamma di frequenze più stretta (risposta in frequenza del nastro).
        'subtractive_on': False,
        'fm_on': False,
        'granular_on': True, 'gran_dens_src': "Variazione Movimento", 'gran_dur_src': "Dettaglio",
        'gran_amp_src': "Movimento", 'gran_pitch_src': "Movimento",
        'gran_dens_min': 3, 'gran_dens_max': 9, 'gran_dur_min': 0.02, 'gran_dur_max': 0.09,
        'gran_amp_min': 0.05, 'gran_amp_max': 0.2, 'gran_pitch_min': 80, 'gran_pitch_max': 1200, 'gran_gain': 1.2,
        'noise_on': True, 'noise_amp_src': "Dettaglio", 'noise_amp_min': 0.05, 'noise_amp_max': 0.25, 'noise_gain': 0.8,
        'glitch_on': True, 'glitch_character': "Pulito / Digitale (repeat + reverse, poco rumore)", 'glitch_factor_src': "Variazione Movimento", 'glitch_intensity_src': "Variazione Movimento",
        'glitch_factor_min': 0.1, 'glitch_factor_max': 0.4, 'glitch_intensity_min': 0.3, 'glitch_intensity_max': 0.7,
        'delay_on': True, 'delay_time_src': "Movimento", 'delay_feedback_src': "Variazione Movimento",
        'delay_time_min': 0.15, 'delay_time_max': 0.4, 'delay_feedback_min': 0.5, 'delay_feedback_max': 0.85,
        'reverb_on': True, 'reverb_decay_src': "Dettaglio", 'reverb_mix_src': "Movimento",
        'reverb_decay_min': 1.5, 'reverb_decay_max': 3.5, 'reverb_mix_min': 0.25, 'reverb_mix_max': 0.5,
        'eq_on': True, 'eq_low_src': "Dettaglio", 'eq_mid_src': "Movimento", 'eq_high_src': "Variazione Movimento",
        'eq_gain_min': -10.0, 'eq_gain_max': 4.0,
        'panning_on': True, 'pan_src': "Variazione Movimento",
    },
    "🎹 Piano Fantasma / E-Piano Ambient": {
        # Layer E-Piano FM come protagonista: rapporto 1.4 (timbro caldo, non metallico — vedi
        # help del rapporto per il riferimento al patch DX7 "E.Piano 1" originale), densità bassa
        # per note isolate e distanziate, decadimento lungo, immerso in riverbero/delay ampi.
        'subtractive_on': False, 'fm_on': False,
        'epiano_on': True, 'epiano_gain': 1.0,
        'epiano_dens_src': "Movimento", 'epiano_pitch_src': "Luminosità",
        'epiano_bright_src': "Dettaglio", 'epiano_amp_src': "Movimento",
        'epiano_dens_min': 0, 'epiano_dens_max': 2, 'epiano_pitch_min': 130, 'epiano_pitch_max': 500,
        'epiano_amp_min': 0.3, 'epiano_amp_max': 0.8, 'epiano_note_dur': 1.6, 'epiano_mod_ratio': 1.4,
        'granular_on': False, 'pluck_on': False, 'noise_on': False, 'glitch_on': False,
        'delay_on': True, 'delay_time_src': "Luminosità", 'delay_feedback_src': "Movimento",
        'delay_time_min': 0.15, 'delay_time_max': 0.35, 'delay_feedback_min': 0.3, 'delay_feedback_max': 0.55,
        'reverb_on': True, 'reverb_decay_src': "Luminosità", 'reverb_mix_src': "Movimento",
        'reverb_decay_min': 3.0, 'reverb_decay_max': 5.0, 'reverb_mix_min': 0.4, 'reverb_mix_max': 0.65,
        'eq_on': False,
        'panning_on': True, 'pan_src': "Centro di Massa Orizzontale",
    },
    "🪕 Arpa Eolica / Corde Sospese": {
        # Layer Corde in modalità "arpa": durezza bassa (pizzicata morbida, non martellata),
        # 3 voci all'unisono con scordatura contenuta per lo scintillio/battimento tipico di
        # un'arpa o di corde multiple che risuonano insieme, densità bassa e registro alto.
        'subtractive_on': False, 'fm_on': False, 'epiano_on': False,
        'pluck_on': True, 'pluck_gain': 1.0,
        'pluck_dens_src': "Movimento", 'pluck_pitch_src': "Luminosità",
        'pluck_damp_src': "Luminosità", 'pluck_amp_src': "Movimento",
        'pluck_dens_min': 0, 'pluck_dens_max': 2, 'pluck_pitch_min': 150, 'pluck_pitch_max': 900,
        'pluck_amp_min': 0.3, 'pluck_amp_max': 0.7, 'pluck_duration': 0.8,
        'pluck_hardness': 0.1, 'pluck_unison_voices': 3, 'pluck_unison_detune': 6.0,
        'granular_on': False, 'noise_on': False, 'glitch_on': False,
        'delay_on': True, 'delay_time_src': "Luminosità", 'delay_feedback_src': "Luminosità",
        'delay_time_min': 0.2, 'delay_time_max': 0.4, 'delay_feedback_min': 0.35, 'delay_feedback_max': 0.6,
        'reverb_on': True, 'reverb_decay_src': "Luminosità", 'reverb_mix_src': "Luminosità",
        'reverb_decay_min': 3.5, 'reverb_decay_max': 5.0, 'reverb_mix_min': 0.5, 'reverb_mix_max': 0.75,
        'eq_on': False,
        'panning_on': True, 'pan_src': "Centro di Massa Orizzontale",
    },
    "🔨 Corde Martellate / Percussione Metallica": {
        # Stesso layer Corde, carattere opposto: durezza alta (eccitazione quasi tutta rumore
        # differenziato, molto più tagliente — vedi hammer_hardness), durata breve e densità più
        # alta per un ritmo percussivo, decadimento rapido (g più basso). Niente riverbero: resta
        # secco e diretto, coerente con un carattere industriale/metallico, non ambientale.
        'subtractive_on': False, 'fm_on': False, 'epiano_on': False,
        'pluck_on': True, 'pluck_gain': 1.1,
        'pluck_dens_src': "Dettaglio", 'pluck_pitch_src': "Movimento",
        'pluck_damp_src': "Dettaglio", 'pluck_amp_src': "Variazione Movimento",
        'pluck_dens_min': 1, 'pluck_dens_max': 4, 'pluck_pitch_min': 80, 'pluck_pitch_max': 600,
        'pluck_amp_min': 0.4, 'pluck_amp_max': 0.9, 'pluck_duration': 0.3,
        'pluck_hardness': 0.85, 'pluck_unison_voices': 2, 'pluck_unison_detune': 12.0,
        'granular_on': False,
        'noise_on': True, 'noise_amp_src': "Variazione Movimento", 'noise_amp_min': 0.02, 'noise_amp_max': 0.1,
        'glitch_on': True, 'glitch_character': "Pulito / Digitale (repeat + reverse, poco rumore)",
        'glitch_factor_src': "Variazione Movimento", 'glitch_intensity_src': "Dettaglio",
        'glitch_factor_min': 0.03, 'glitch_factor_max': 0.15, 'glitch_intensity_min': 0.2, 'glitch_intensity_max': 0.5,
        'delay_on': True, 'delay_time_src': "Dettaglio", 'delay_feedback_src': "Dettaglio",
        'delay_time_min': 0.04, 'delay_time_max': 0.1, 'delay_feedback_min': 0.15, 'delay_feedback_max': 0.35,
        'reverb_on': False,
        'eq_on': True, 'eq_low_src': "Movimento", 'eq_mid_src': "Dettaglio", 'eq_high_src': "Dettaglio",
        'eq_gain_min': -6.0, 'eq_gain_max': 8.0,
        'panning_on': True, 'pan_src': "Densità Contorni",
    },
}

def scale_frequency_exponential(data_raw: list, freq_min: float, freq_max: float) -> list:
    """Mappa una serie di valori di controllo (es. luminosità/movimento frame-per-frame) su un
    range di frequenza in scala ESPONENZIALE (log2 in Hz) invece che lineare — lo standard
    professionale '1V/ottava' usato in ogni sintetizzatore analogico o digitale.

    Perché: l'orecchio percepisce l'altezza in modo logaritmico. Un raddoppio di frequenza è
    sempre 'un'ottava' percepita, che si parta da 55Hz o da 880Hz. Con un mapping lineare in Hz,
    lo stesso range assoluto (es. 80-400Hz) risulta percettivamente compresso nella parte bassa
    (80→120Hz è quasi un'ottava e mezza) ed espanso in quella alta (280→320Hz è un intervallo
    minuscolo), anche se il video si muove in modo uniforme. Con lo scaling esponenziale, passi
    uguali del segnale di controllo producono sempre lo stesso salto musicale percepito, ovunque
    nel range — esattamente il motivo per cui i synth modulari usano l'esponenziale per il
    controllo di pitch/cutoff invece del semplice Hz/volt lineare."""
    if not data_raw:
        return []
    lo, hi = min(data_raw), max(data_raw)
    log_min = np.log2(max(freq_min, 1e-6))
    log_max = np.log2(max(freq_max, 1e-6))
    normalized_log = np.interp(data_raw, (lo, hi), (log_min, log_max))
    return np.exp2(normalized_log).tolist()

# ── QUANTIZZAZIONE A SCALA MUSICALE ─────────────────────────────────────────
# Intervalli in semitoni dalla fondamentale. La pentatonica è la scelta classica per musica
# guidata da dati esterni imprevedibili (usata da Brian Eno nei suoi sistemi generativi, e in
# progetti di data-sonification): qualsiasi combinazione di note pentatoniche resta consonante,
# quindi il video può "suonare" la melodia senza mai produrre un intervallo dissonante.
SCALE_NONE = "Nessuna (frequenza libera/continua)"
MUSICAL_SCALES = {
    "Pentatonica Maggiore": [0, 2, 4, 7, 9],
    "Pentatonica Minore": [0, 3, 5, 7, 10],
    "Maggiore (Ionica)": [0, 2, 4, 5, 7, 9, 11],
    "Minore Naturale (Eolia)": [0, 2, 3, 5, 7, 8, 10],
    "Minore Armonica": [0, 2, 3, 5, 7, 8, 11],
    "Dorica": [0, 2, 3, 5, 7, 9, 10],
    "Frigia": [0, 1, 3, 5, 7, 8, 10],
    "Cromatica (tutti i semitoni)": list(range(12)),
}

def quantize_to_scale(freqs: list, root_freq: float, scale_name: str) -> list:
    """Agganciano ('quantizzano') una serie di frequenze continue alla nota più vicina di una
    scala musicale, invece di lasciarle libere di cadere ovunque nello spettro (microtonali).
    Senza questo passaggio ogni 'nota' avrebbe comunque una sua frequenza distinta guidata dal
    video, ma il risultato suonerebbe come un glissando continuo, non come una melodia
    riconoscibile — è la quantizzazione a rendere l'altezza discreta e musicale."""
    if not freqs or scale_name == SCALE_NONE or scale_name not in MUSICAL_SCALES:
        return freqs

    intervals = MUSICAL_SCALES[scale_name]
    # Costruisce tutte le frequenze della scala su un ampio range di ottave (dal sub-basso
    # all'acuto) così la ricerca del vicino più prossimo funziona per qualsiasi frequenza in
    # ingresso, senza dover gestire manualmente i casi limite di ottava/wrap-around.
    scale_freqs = sorted(
        root_freq * (2.0 ** (octave_n + interval / 12.0))
        for octave_n in range(-6, 7)
        for interval in intervals
    )
    scale_freqs_arr = np.array(scale_freqs)
    freqs_arr = np.array(freqs)

    idx = np.searchsorted(scale_freqs_arr, freqs_arr)
    idx = np.clip(idx, 1, len(scale_freqs_arr) - 1)
    left = scale_freqs_arr[idx - 1]
    right = scale_freqs_arr[idx]
    quantized = np.where(np.abs(freqs_arr - left) <= np.abs(freqs_arr - right), left, right)
    return quantized.tolist()

def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è installato e disponibile nel PATH."""
    return shutil.which("ffmpeg") is not None

def cleanup_session_temp_files(session_id: str) -> None:
    """Rimuove eventuali file temporanei residui di una sessione precedente
    (es. se l'utente carica un nuovo video senza aver completato la generazione
    precedente). Ogni file temporaneo della sessione contiene session_id nel nome,
    quindi non tocca mai i file di altre sessioni/utenti concorrenti."""
    patterns = [
        f"temp_input_{session_id}_*",
        f"output_audio_{session_id}.wav",
        f"preview_audio_{session_id}.wav",
        f"temp_original_audio_{session_id}.aac",
        f"output_*_{session_id}_*.mp4",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass

def preset_to_filename_slug(preset_name: str) -> str:
    """Converte il nome di un preset in uno slug sicuro da usare nei nomi file
    (rimuove emoji/simboli, sostituisce spazi e slash con underscore)."""
    if not preset_name or preset_name == PRESET_MANUALE:
        return ""
    # Rimuove tutto ciò che non è lettera, numero, spazio o underscore (elimina emoji/simboli)
    cleaned = re.sub(r"[^\w\s]", "", preset_name, flags=re.UNICODE)
    cleaned = cleaned.strip().replace(" ", "_").replace("__", "_")
    return cleaned.strip("_")

def validate_video_file(uploaded_file) -> bool:
    """Valida le dimensioni del file video caricato."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ Il file è troppo grande. Dimensione massima consentita: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB.")
        return False
    return True

def analyze_video_frames(video_path: str) -> Tuple[list, list, list, list, list, list, list, list, float, float]:
    """
    Analizza i frame di un video per estrarre dati visivi.

    Args:
        video_path (str): Il percorso del file video da analizzare.

    Returns:
        Tuple[list, list, list, list, list, list, list, list, float, float]: Una tupla contenente liste di dati
        per luminosità, dettaglio, movimento, variazione del movimento, centro di massa orizzontale,
        centro di massa verticale, densità contorni, variazione colore, la durata effettiva del video in secondi,
        e il frame rate (FPS) del video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Impossibile aprire il video: {video_path}")
        return [], [], [], [], [], [], [], [], 0.0, 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or frame_count <= 0:
        st.error("❌ Impossibile leggere frame rate/numero di frame dal video. Il file potrebbe essere corrotto o in un formato non supportato.")
        cap.release()
        return [], [], [], [], [], [], [], [], 0.0, 0.0

    duration_seconds = frame_count / fps

    if duration_seconds > MAX_DURATION:
        st.error(f"❌ Video troppo lungo. Durata massima consentita: {MAX_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], [], [], [], 0.0, 0.0
    if duration_seconds < MIN_DURATION:
        st.error(f"❌ Video troppo corto. Durata minima consentita: {MIN_DURATION} secondi. Il tuo video è di {duration_seconds:.2f} secondi.")
        cap.release()
        return [], [], [], [], [], [], [], [], 0.0, 0.0

    luminosity_data = []
    detail_data = [] # Misurato come deviazione standard dell'intensità dei pixel
    movement_data = [] # Differenza assoluta media tra frame consecutivi
    variation_movement_data = [] # Variazione del movimento
    horizontal_mass_center_data = [] # Centro di massa orizzontale per il panning
    vertical_mass_center_data = [] # Centro di massa verticale per la simulazione di altezza (su/giù)
    edge_density_data = [] # Densità dei contorni (Sobel) - per pattern/linee/geometrie
    color_variation_data = [] # Deviazione standard della tonalità (Hue) - per varietà cromatica

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
            cy = int(M['m01'] / M['m00'])
            vertical_mass_center_data.append(cy / frame.shape[0]) # 0=alto del frame, 1=basso del frame
        else:
            horizontal_mass_center_data.append(0.5) # Centro se il frame è vuoto o scuro
            vertical_mass_center_data.append(0.5)

        # Densità contorni (magnitudine media del gradiente Sobel) - utile per pattern/linee/geometrie
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(edge_magnitude) / 255.0 # Normalizzato approssimativamente tra 0 e 1
        edge_density_data.append(min(edge_density, 1.0))

        # Variazione colore (deviazione standard della tonalità/Hue) - utile per varietà cromatica
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_frame[:, :, 0]
        color_variation = np.std(hue_channel) / 180.0 # Hue in OpenCV va da 0 a 179
        color_variation_data.append(min(color_variation, 1.0))

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
    for arr in [detail_data, movement_data, variation_movement_data, horizontal_mass_center_data,
                vertical_mass_center_data, edge_density_data, color_variation_data]:
        while len(arr) < max_len:
            arr.append(arr[-1] if arr else 0.0) # Aggiunge l'ultimo valore o 0.0 se vuoto

    gc.collect() # Libera memoria

    return (luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data,
            vertical_mass_center_data, edge_density_data, color_variation_data, duration_seconds, fps)


@st.cache_data(show_spinner="🎞️ Analisi del video in corso...")
def analyze_video_frames_cached(video_path: str, file_hash: str) -> Tuple[list, list, list, list, list, list, list, list, float, float]:
    """Wrapper cache di analyze_video_frames. Senza questa cache, Streamlit riesegue l'intero
    script (e quindi anche l'analisi frame-per-frame del video, che decodifica ogni singolo
    fotogramma con OpenCV) ad OGNI interazione con qualsiasi controllo dell'interfaccia — non solo
    quando serve, ma anche solo spuntando una checkbox o spostando uno slider. Per un video di
    qualche minuto questo significa ricalcolare tutto da capo più volte al secondo mentre l'utente
    regola i parametri, rendendo l'interfaccia lenta e frustrante.

    file_hash (non usato nel corpo della funzione) serve solo a rendere la chiave di cache dipendente
    dal CONTENUTO del file, non solo dal suo path: il path include il nome del file originale, quindi
    se un utente carica due video diversi con lo stesso nome nella stessa sessione, senza l'hash la
    cache riuserebbe erroneamente l'analisi del primo video anche per il secondo."""
    return analyze_video_frames(video_path)


def generate_synthetic_feature_curves(duration_seconds: float, fps: float = 30.0, seed: int = None
                                       ) -> Tuple[list, list, list, list, list, list, list, list, float, float]:
    """Genera 8 curve di controllo sintetiche al posto di quelle estratte da un video, per usare
    VideoSound-Gen come strumento generativo standalone senza caricare alcun file. Restituisce
    esattamente lo stesso formato di analyze_video_frames_cached (8 liste + durata + fps), quindi
    tutto il resto della pipeline — layer, effetti, preset, Melodia — funziona invariato: non sa
    (e non deve sapere) se sta 'guardando' un video vero o una traiettoria generata.

    Ogni curva è una somma di 2-3 sinusoidi lente a frequenze/fasi indipendenti (diverse per ogni
    feature, così luminosità/movimento/ecc. non si muovono mai in perfetto lock-step, che
    suonerebbe artificiale) più un filo di rumore smussato, per un movimento organico e mai
    esattamente periodico — lo stesso principio degli LFO multipli usati nei synth modulari."""
    rng = np.random.RandomState(seed)
    n_frames = max(2, int(duration_seconds * fps))
    t = np.linspace(0, duration_seconds, n_frames)

    def organic_curve(base_freq_hz: float, n_harmonics: int = 3, noise_amount: float = 0.15) -> list:
        curve = np.zeros(n_frames)
        for h in range(1, n_harmonics + 1):
            freq = base_freq_hz * h * (0.7 + 0.6 * rng.rand())
            phase = rng.uniform(0, 2 * np.pi)
            curve += np.sin(2 * np.pi * freq * t + phase) / h
        noise = rng.randn(n_frames)
        smoothing_kernel = np.ones(5) / 5.0
        noise = np.convolve(noise, smoothing_kernel, mode='same')
        curve = curve + noise_amount * noise
        curve = curve - curve.min()
        if curve.max() > 1e-9:
            curve = curve / curve.max()
        return curve.tolist()

    # Frequenze base diverse per ciascuna feature: più basse per quelle che nel video reale
    # tendono a cambiare lentamente (luminosità, posizione), più alte per quelle più "nervose"
    # (movimento, variazione movimento).
    luminosity_data = organic_curve(0.05)
    detail_data = organic_curve(0.08)
    movement_data = organic_curve(0.12)
    variation_movement_data = organic_curve(0.15)
    horizontal_mass_center_data = organic_curve(0.03)
    vertical_mass_center_data = organic_curve(0.04)
    edge_density_data = organic_curve(0.07)
    color_variation_data = organic_curve(0.06)

    return (luminosity_data, detail_data, movement_data, variation_movement_data,
            horizontal_mass_center_data, vertical_mass_center_data, edge_density_data,
            color_variation_data, duration_seconds, fps)


class AudioGenerator:
    def __init__(self, sample_rate: int, total_duration_seconds: float):
        self.sample_rate = sample_rate
        self.total_duration_seconds = total_duration_seconds
        self.total_samples = int(self.total_duration_seconds * self.sample_rate)
        self.time_array = np.linspace(0, self.total_duration_seconds, self.total_samples, endpoint=False)


    def _interp_data_to_audio_length(self, data_per_frame: list) -> np.ndarray:
        """Interpola i dati per frame alla lunghezza dell'array audio."""
        if len(data_per_frame) == 0:
            return np.zeros(self.total_samples)
        
        original_time_points = np.linspace(0, self.total_duration_seconds, len(data_per_frame), endpoint=True)
        return np.interp(self.time_array, original_time_points, data_per_frame)

    @staticmethod
    def _poly_blep_vectorized(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Correzione PolyBLEP (Polynomial Band-Limited Step), vettorizzata su tutto l'array
        invece che campione per campione: attenua l'aliasing nelle discontinuità di square/
        sawtooth senza un loop Python. t = fase normalizzata in [0,1), dt = frequenza normalizzata
        (freq/sample_rate) nello stesso punto. Standard nella sintesi digitale di forme d'onda."""
        dt_safe = np.maximum(dt, 1e-12)
        y = np.zeros_like(t)

        mask1 = t < dt
        tt = np.where(mask1, t / dt_safe, 0.0)
        y = np.where(mask1, tt + tt - tt * tt - 1.0, y)

        mask2 = t > (1.0 - dt)
        tt2 = np.where(mask2, (t - 1.0) / dt_safe, 0.0)
        y = np.where(mask2, tt2 * tt2 + tt2 + tt2 + 1.0, y)

        return y

    def apply_resonant_filter(self, audio_array: np.ndarray, cutoff_data: list, resonance_data: list) -> np.ndarray:
        """Filtro passa-basso risonante (biquad, formule RBJ Audio EQ Cookbook) con cutoff e
        risonanza (Q) modulati dinamicamente nel tempo. Questa è la parte 'sottrattiva' vera
        della sintesi sottrattiva: scolpisce le armoniche di un'onda già generata (tipicamente
        square/sawtooth, ricche di armoniche) invece di limitarsi a modularne ampiezza e
        frequenza fondamentale come faceva prima. Processato a blocchi, stesso schema già usato
        per delay/reverb: i coefficienti sono ricalcolati periodicamente e lo stato interno del
        filtro (zi) continua da un blocco al successivo, per evitare click alle giunzioni."""
        n_samples = len(audio_array)
        if n_samples == 0 or len(cutoff_data) == 0:
            return audio_array

        nyquist = 0.5 * self.sample_rate

        # Aggiorna i coefficienti ogni ~20ms: abbastanza spesso da seguire la modulazione video,
        # abbastanza raro da restare economico (stesso compromesso di delay/reverb).
        chunk_size = max(1, int(0.02 * self.sample_rate))
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        chunk_start_indices = np.minimum(np.arange(n_chunks) * chunk_size, n_samples - 1)
        chunk_times = self.time_array[chunk_start_indices]

        cutoff_time_points = np.linspace(0, self.total_duration_seconds, len(cutoff_data), endpoint=True)
        resonance_time_points = np.linspace(0, self.total_duration_seconds, len(resonance_data), endpoint=True)
        # Il cutoff non può mai raggiungere/superare Nyquist (coefficienti non validi): tetto di
        # sicurezza al 95% di Nyquist. Q limitato per evitare risonanze instabili (auto-oscillazione).
        cutoff_at_chunk = np.clip(np.interp(chunk_times, cutoff_time_points, cutoff_data), 20.0, nyquist * 0.95)
        resonance_at_chunk = np.clip(np.interp(chunk_times, resonance_time_points, resonance_data), 0.5, 10.0)

        w0 = 2 * np.pi * cutoff_at_chunk / self.sample_rate
        alpha = np.sin(w0) / (2 * resonance_at_chunk)
        cosw0 = np.cos(w0)

        b0 = (1 - cosw0) / 2
        b1 = 1 - cosw0
        b2 = (1 - cosw0) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw0
        a2 = 1 - alpha

        filtered = np.empty_like(audio_array, dtype=np.float64)
        zi = np.zeros(2)  # stato del biquad (2 poli), continua tra un blocco e il successivo

        idx = 0
        chunk_idx = 0
        while idx < n_samples:
            end = min(idx + chunk_size, n_samples)
            b = np.array([b0[chunk_idx], b1[chunk_idx], b2[chunk_idx]]) / a0[chunk_idx]
            a = np.array([1.0, a1[chunk_idx] / a0[chunk_idx], a2[chunk_idx] / a0[chunk_idx]])

            out_chunk, zi = lfilter(b, a, audio_array[idx:end], zi=zi)
            filtered[idx:end] = out_chunk

            idx = end
            chunk_idx += 1

        return filtered

    def generate_subtractive_waveform(self, freq_data: list, amp_data: list, waveform_type: str = "sine",
                                       band_limited: bool = False) -> np.ndarray:
        """Genera una forma d'onda sottrattiva base con frequenza e ampiezza dinamiche.

        band_limited: square e sawtooth generati "al naturale" (np.sign/rampa lineare) aliasano
        pesantemente alle frequenze medio-alte, perché contengono un salto discontinuo che il
        campionamento digitale non può rappresentare correttamente. Questo è storicamente proprio
        il tipo di artefatto ricercato in molta musica glitch/lo-fi, quindi resta il default
        (band_limited=False). Se True, applica una correzione PolyBLEP alle discontinuità per
        un suono più "pulito" in stile synth analogico/digitale professionale: è una scelta
        estetica esplicita, non una correzione "giusta" da applicare sempre."""
        freq_interp = self._interp_data_to_audio_length(freq_data)
        amp_interp = self._interp_data_to_audio_length(amp_data)

        audio = np.zeros(self.total_samples)
        phase_increment = 2 * np.pi * freq_interp / self.sample_rate
        phase = np.cumsum(phase_increment)

        if waveform_type == "sine":
            waveform = np.sin(phase)
        elif waveform_type == "square":
            normalized_phase = (phase / (2 * np.pi)) % 1.0
            waveform = np.where(normalized_phase < 0.5, 1.0, -1.0)
            if band_limited:
                dt = phase_increment / (2 * np.pi)
                waveform = (waveform
                            + self._poly_blep_vectorized(normalized_phase, dt)
                            - self._poly_blep_vectorized((normalized_phase - 0.5) % 1.0, dt))
        elif waveform_type == "sawtooth":
            normalized_phase = (phase / (2 * np.pi)) % 1.0
            waveform = 2 * (normalized_phase - 0.5)
            if band_limited:
                dt = phase_increment / (2 * np.pi)
                waveform = waveform - self._poly_blep_vectorized(normalized_phase, dt)
        else: # default to sine
            waveform = np.sin(phase)

        audio = waveform * amp_interp
        return audio

    def generate_fm_epiano_layer(self, density_data: list, pitch_data: list, brightness_data: list, amp_data: list,
                                  note_duration: float = 1.2, mod_ratio: float = 1.0,
                                  trigger_window_seconds: float = 0.2) -> np.ndarray:
        """Layer 'e-piano' FM (stile DX7/Rhodes): a differenza del layer FM continuo esistente,
        qui la sintesi è per NOTE discrete con inviluppo. Il dettaglio che fa davvero la
        differenza timbrica: l'indice di modulazione ha il suo inviluppo separato, che decade
        PIÙ VELOCE di quello dell'ampiezza. All'attacco il suono è ricco di armoniche (indice
        alto → timbro 'a campana'/metallico), poi mentre l'ampiezza è ancora sostenuta il suono
        si 'ammorbidisce' verso una sinusoide quasi pura: è esattamente questo doppio decadimento
        indipendente a dare il carattere riconoscibile degli e-piano FM classici, non solo
        l'inviluppo di volume. Le note sono innescate su finestre temporali fisse (stesso motivo
        già spiegato per il layer Corde: note più fitte di ~200ms si fonderebbero comunque)."""
        audio = np.zeros(self.total_samples)
        n_frames = len(density_data)

        if n_frames == 0:
            return audio

        note_duration_samples = max(64, int(note_duration * self.sample_rate))
        window_samples = max(1, int(trigger_window_seconds * self.sample_rate))
        n_windows = max(1, (self.total_samples + window_samples - 1) // window_samples)

        original_time_points = np.linspace(0, self.total_duration_seconds, n_frames, endpoint=True)
        window_indices = np.minimum(np.arange(n_windows) * window_samples, self.total_samples - 1)
        window_times = self.time_array[window_indices]

        density_at_window = np.interp(window_times, original_time_points, density_data)
        pitch_at_window = np.interp(window_times, original_time_points, pitch_data)
        brightness_at_window = np.interp(window_times, original_time_points, brightness_data)
        amp_at_window = np.interp(window_times, original_time_points, amp_data)

        # Asse temporale della nota, condiviso (la forma dell'inviluppo dipende solo dal tempo
        # trascorso dall'attacco, non dalla posizione assoluta nel brano).
        t = np.arange(note_duration_samples) / self.sample_rate
        attack_samples = max(1, int(0.005 * self.sample_rate))  # attacco percussivo, ~5ms
        attack_ramp = np.linspace(0.0, 1.0, attack_samples)

        amp_decay_tc = 1.2   # costante di tempo (sec) del decadimento d'ampiezza
        mod_decay_tc = 0.25  # più corto: l'indice di modulazione (la "brillantezza") si smorza prima

        base_amp_env = np.ones(note_duration_samples)
        base_amp_env[:attack_samples] = attack_ramp
        base_amp_env *= np.exp(-t / amp_decay_tc)
        base_mod_env_shape = np.exp(-t / mod_decay_tc)  # moltiplicato poi per l'indice iniziale per-nota

        for i in range(n_windows):
            num_notes = int(density_at_window[i])
            if num_notes == 0:
                continue

            start_w = i * window_samples
            end_w = min(start_w + window_samples, self.total_samples)
            if start_w >= end_w:
                continue

            current_pitch = max(20.0, float(pitch_at_window[i]))
            current_brightness = float(np.clip(brightness_at_window[i], 0.0, 1.0))
            current_amp = float(amp_at_window[i])

            start_notes = np.random.randint(start_w, end_w, size=num_notes)
            jitter = (np.random.rand(num_notes) - 0.5) * 0.02  # +/-1%: intonazione naturale, non stonata come le corde

            # Indice di modulazione iniziale: range ~0.5 (timbro morbido) - 6.0 (timbro brillante/metallico)
            mod_index0 = 0.5 + 5.5 * current_brightness
            mod_index_env = mod_index0 * base_mod_env_shape

            for k in range(num_notes):
                freq = max(20.0, current_pitch * (1.0 + jitter[k]))
                carrier_freq = freq
                modulator_freq = freq * mod_ratio

                modulator_phase = 2 * np.pi * modulator_freq * t
                carrier_phase = 2 * np.pi * carrier_freq * t + mod_index_env * np.sin(modulator_phase)
                note_signal = np.sin(carrier_phase) * base_amp_env * current_amp * 0.3

                start = start_notes[k]
                end = min(start + note_duration_samples, self.total_samples)
                audio[start:end] += note_signal[:end - start]

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

    def _karplus_strong_pluck(self, D: int, duration_samples: int, g: float, hammer_hardness: float = 0.0) -> np.ndarray:
        """Calcola una singola pizzicata Karplus-Strong: y[n] = eccitazione[n] per n<D (lunga
        quanto la linea di ritardo), poi y[n] = 0.5*g*(y[n-D] + y[n-D-1]) per n>=D.

        Questa è la stessa equazione IIR di prima, ma calcolata a blocchi di D campioni invece che
        con scipy.signal.lfilter: lfilter con un array 'a' lungo e quasi tutto zero elabora ogni
        campione di output usando TUTTI i coefficienti (costo O(durata * D) per pizzicata — con
        migliaia di pizzicate su un video lungo, minuti di attesa, verificato con un benchmark).
        Ogni blocco di D campioni qui legge solo dal blocco precedente (mai da se stesso), quindi
        è vettorizzabile in un colpo solo con NumPy: costo O(durata) per pizzicata, indipendente da D.

        hammer_hardness (0-1): l'eccitazione classica di Karplus-Strong è rumore bianco puro
        (carattere di corda pizzicata, morbido). Miscelando la sua derivata prima (differenza tra
        campioni consecutivi, che esalta le componenti ad alta frequenza) si ottiene un transiente
        più duro e percussivo — più vicino a un martelletto di pianoforte/clavicembalo che a un dito
        che pizzica una corda. hardness=0 → comportamento originale invariato."""
        # y_ext[0] rappresenta y[-1] = 0 (silenzio prima dell'eccitazione); serve per evitare
        # indici negativi quando il primo blocco legge un campione prima dell'inizio del rumore.
        y_ext = np.zeros(duration_samples + 1)
        burst_len = min(D, duration_samples)
        noise_burst = np.random.uniform(-1.0, 1.0, burst_len)

        if hammer_hardness > 0.0:
            hard_burst = np.diff(noise_burst, prepend=0.0)
            # np.diff raddoppia grosso modo l'ampiezza media (differenza di due valori indipendenti
            # in [-1,1]): riscalata per restare paragonabile in livello al rumore bianco puro.
            hard_burst = hard_burst / 1.6
            excitation = (1.0 - hammer_hardness) * noise_burst + hammer_hardness * hard_burst
        else:
            excitation = noise_burst

        y_ext[1:1 + burst_len] = excitation

        pos = D
        while pos < duration_samples:
            block_len = min(D, duration_samples - pos)
            ext_idx = np.arange(pos, pos + block_len) + 1
            y_ext[ext_idx] = 0.5 * g * (y_ext[ext_idx - D] + y_ext[ext_idx - D - 1])
            pos += block_len

        return y_ext[1:]

    def generate_pluck_layer(self, density_data: list, pitch_data: list, damping_data: list, amp_data: list,
                              max_pluck_duration: float = 0.4, trigger_window_seconds: float = 0.15,
                              hammer_hardness: float = 0.0, unison_voices: int = 1, unison_detune_cents: float = 0.0) -> np.ndarray:
        """Layer a modellazione fisica (Karplus-Strong): sintetizza 'pizzicate' di corda invece di
        onde/grani. Algoritmo classico: rumore bianco che eccita una linea di ritardo di lunghezza
        D = sample_rate/pitch, richiusa in retroazione con un filtro passa-basso a 2 campioni
        (media mobile), che simula lo smorzamento naturale di una corda che vibra.

        y[n] = x[n] + g * 0.5 * (y[n-D] + y[n-D-1])

        Questa è un'equazione alle differenze IIR standard, calcolata da _karplus_strong_pluck
        a blocchi vettorizzati con NumPy (vedi il commento lì per il perché non usiamo né un loop
        Python campione-per-campione né scipy.signal.lfilter direttamente).

        A differenza del layer granulare, qui le pizzicate sono innescate su finestre temporali
        fisse (trigger_window_seconds) invece che una per ogni fotogramma video: pizzicate più
        fitte di ~100-150ms diventerebbero comunque indistinguibili tra loro (perderebbero il
        carattere "corda pizzicata" per fondersi in una texture continua, che è già il ruolo del
        layer granulare), quindi agganciarle al frame rate del video non aggiungerebbe nulla di
        espressivo — solo molte più pizzicate da calcolare per un video lungo e denso.

        hammer_hardness (0-1): vedi _karplus_strong_pluck — 0 è la pizzicata originale (rumore
        bianco puro), valori più alti danno un attacco più duro/percussivo (stile martelletto).

        unison_voices (1-3) + unison_detune_cents: invece di una sola linea di ritardo per nota,
        ne somma 2-3 leggermente scordate tra loro (come le corde multiple all'unisono di un
        pianoforte vero), producendo il caratteristico 'battimento' che dà corpo e movimento al
        suono invece di una pizzicata singola e statica. unison_voices=1 disattiva l'effetto
        (comportamento identico a prima)."""
        audio = np.zeros(self.total_samples)
        n_frames = len(density_data)

        if n_frames == 0:
            return audio

        unison_voices = max(1, min(3, int(unison_voices)))

        pluck_duration_samples = max(64, int(max_pluck_duration * self.sample_rate))
        window_samples = max(1, int(trigger_window_seconds * self.sample_rate))
        n_windows = max(1, (self.total_samples + window_samples - 1) // window_samples)

        # Stesso principio già applicato a granulare/glitch/delay/riverbero: si interpola solo
        # nei punti che il loop legge davvero (un valore per finestra), non sull'intera lunghezza
        # audio a risoluzione campione.
        original_time_points = np.linspace(0, self.total_duration_seconds, n_frames, endpoint=True)
        window_indices = np.minimum(np.arange(n_windows) * window_samples, self.total_samples - 1)
        window_times = self.time_array[window_indices]

        density_at_window = np.interp(window_times, original_time_points, density_data)
        pitch_at_window = np.interp(window_times, original_time_points, pitch_data)
        damping_at_window = np.interp(window_times, original_time_points, damping_data)
        amp_at_window = np.interp(window_times, original_time_points, amp_data)

        # Offset di intonazione (in rapporto di frequenza) per ciascuna voce all'unisono, centrati
        # intorno a 0: con 1 voce non c'è offset, con 2-3 sono simmetrici intorno all'intonazione
        # nominale (es. 3 voci a +/-detune e una centrale).
        if unison_voices == 1:
            voice_offsets_cents = np.array([0.0])
        elif unison_voices == 2:
            voice_offsets_cents = np.array([-unison_detune_cents / 2, unison_detune_cents / 2])
        else:
            voice_offsets_cents = np.array([-unison_detune_cents, 0.0, unison_detune_cents])
        voice_ratios = 2.0 ** (voice_offsets_cents / 1200.0)

        for i in range(n_windows):
            num_plucks_in_window = int(density_at_window[i])
            if num_plucks_in_window == 0:
                continue

            start_sample_window = i * window_samples
            end_sample_window = min(start_sample_window + window_samples, self.total_samples)
            if start_sample_window >= end_sample_window:
                continue

            current_pitch = max(20.0, float(pitch_at_window[i]))
            # damping_at_window è in [0,1] guidato dal video: mappato al guadagno di
            # retroazione g (0.90 = smorzamento rapido/percussivo, 0.999 = corda che risuona a lungo)
            g = 0.90 + 0.099 * float(np.clip(damping_at_window[i], 0.0, 1.0))
            current_amp = float(amp_at_window[i])

            start_pluck_samples = np.random.randint(start_sample_window, end_sample_window, size=num_plucks_in_window)
            jitter = (np.random.rand(num_plucks_in_window) - 0.5) * 0.1  # +/-5% intonazione, meno del layer granulare (le corde restano più "accordate")

            for p in range(num_plucks_in_window):
                pluck_freq = max(20.0, current_pitch * (1.0 + jitter[p]))

                pluck_audio = np.zeros(pluck_duration_samples)
                for ratio in voice_ratios:
                    voice_freq = max(20.0, pluck_freq * ratio)
                    D = max(2, int(round(self.sample_rate / voice_freq)))
                    pluck_audio += self._karplus_strong_pluck(D, pluck_duration_samples, g, hammer_hardness=hammer_hardness)
                pluck_audio = pluck_audio / np.sqrt(unison_voices) * current_amp * 0.15  # /sqrt(N): somma voci scorrelate, non /N che sarebbe troppo silenzioso

                start = start_pluck_samples[p]
                end = min(start + pluck_duration_samples, self.total_samples)
                audio[start:end] += pluck_audio[:end - start]

        return audio

    def generate_granular_layer(self, density_data: list, grain_duration_data: list, amp_data: list, pitch_data: list) -> np.ndarray:
        """Genera un layer di sintesi granulare. L'intonazione dei grani è guidata da pitch_data
        (con un piccolo jitter casuale per mantenere una texture organica), non più puramente random."""
        n_frames = len(density_data)
        audio = np.zeros(self.total_samples)

        if n_frames == 0:
            return audio

        samples_per_virtual_frame = int(self.total_samples / n_frames)

        # I 4 parametri sono definiti un valore per frame video, e il loop sotto ne legge
        # comunque un solo valore per segmento (current_density = ...[i * samples_per_virtual_frame]).
        # Interpolarli sull'intera lunghezza dell'audio (milioni di campioni, via
        # _interp_data_to_audio_length) per poi usarne solo ~n_frames valori è il costo reale
        # di questa funzione (confermato via profiling: >80% del tempo totale). Si interpola
        # quindi solo nei punti temporali effettivamente letti dal loop: stessi identici valori,
        # perché np.interp è valutato negli stessi istanti, ma senza calcolare il resto dell'array.
        original_time_points = np.linspace(0, self.total_duration_seconds, n_frames, endpoint=True)
        segment_indices = np.minimum(np.arange(n_frames) * samples_per_virtual_frame, self.total_samples - 1)
        segment_times = self.time_array[segment_indices]

        density_at_segment = np.interp(segment_times, original_time_points, density_data)
        grain_dur_at_segment = np.interp(segment_times, original_time_points, grain_duration_data)
        amp_at_segment = np.interp(segment_times, original_time_points, amp_data)
        pitch_at_segment = np.interp(segment_times, original_time_points, pitch_data)

        for i in range(n_frames):
            current_density = density_at_segment[i]
            current_grain_dur_seconds = grain_dur_at_segment[i]
            current_amp = amp_at_segment[i]
            current_pitch = pitch_at_segment[i]

            num_grains_in_segment = int(current_density)

            if num_grains_in_segment == 0:
                continue

            grain_dur_samples = int(current_grain_dur_seconds * self.sample_rate)
            grain_dur_samples = max(10, grain_dur_samples)

            start_sample_segment = i * samples_per_virtual_frame
            end_sample_segment = min((i + 1) * samples_per_virtual_frame, self.total_samples)

            if start_sample_segment >= end_sample_segment - grain_dur_samples:
                continue

            # Genera in blocco i parametri casuali di tutti i grani del segmento (posizione
            # di partenza e jitter di intonazione) con un'unica chiamata numpy invece di una
            # per grano: piccola ottimizzazione aggiuntiva, non il costo principale.
            max_start = end_sample_segment - grain_dur_samples
            start_grain_samples = np.random.randint(start_sample_segment, max_start, size=num_grains_in_segment)
            jitter = (np.random.rand(num_grains_in_segment) - 0.5) * 0.2
            grain_freqs = np.maximum(20.0, current_pitch * (1.0 + jitter))

            grain_t = np.arange(grain_dur_samples) / self.sample_rate
            hanning_window = np.hanning(grain_dur_samples)

            for g in range(num_grains_in_segment):
                grain_waveform = np.sin(2 * np.pi * grain_freqs[g] * grain_t)
                grain_with_envelope = grain_waveform * hanning_window * current_amp * 0.1

                start_grain_sample = start_grain_samples[g]
                end_grain_sample = start_grain_sample + grain_dur_samples
                audio[start_grain_sample:end_grain_sample] += grain_with_envelope

        return audio

    def add_noise_layer(self, noise_amp_data: list) -> np.ndarray:
        """Genera un layer di rumore modulato (da aggiungere una sola volta all'audio esistente)."""
        noise_amp_interp = self._interp_data_to_audio_length(noise_amp_data)
        noise_layer = np.random.normal(0, 1, self.total_samples) * noise_amp_interp * 0.2
        return noise_layer

    def apply_glitch_effect(self, audio_array: np.ndarray, glitch_factor_data: list, glitch_intensity_data: list,
                             type_weights: dict = None) -> np.ndarray:
        """Applica un effetto glitch all'audio.

        type_weights: dict opzionale tipo {"repeat": 0.4, "noise": 0.2, "reverse": 0.4} per pesare
        la scelta del tipo di glitch invece di sceglierlo sempre in modo uniforme/casuale. Se None,
        i tre tipi hanno la stessa probabilità (comportamento precedente)."""
        glitched_audio = np.copy(audio_array)
        glitched_audio = np.nan_to_num(glitched_audio, nan=0.0)

        # Il while loop qui sotto legge il valore interpolato in poche migliaia di indici al
        # massimo (avanza di ~0.1s per volta), non in ogni campione. Precalcolare due array
        # interpolati sull'intera lunghezza dell'audio (milioni di campioni) per poi leggerne
        # solo una piccola frazione è lavoro sprecato: si interpola quindi punto per punto,
        # esattamente negli istanti effettivamente visitati dal loop (stesso identico valore
        # che si otterrebbe leggendo l'array precalcolato allo stesso indice).
        glitch_factor_time_points = np.linspace(0, self.total_duration_seconds, len(glitch_factor_data), endpoint=True)
        glitch_intensity_time_points = np.linspace(0, self.total_duration_seconds, len(glitch_intensity_data), endpoint=True)

        glitch_check_interval_samples = int(0.1 * self.sample_rate) # Controlla ogni 100ms

        if type_weights is None:
            type_names = ["repeat", "noise", "reverse"]
            type_probs = [1/3, 1/3, 1/3]
        else:
            type_names = list(type_weights.keys())
            raw_weights = np.array(list(type_weights.values()), dtype=np.float64)
            type_probs = (raw_weights / raw_weights.sum()).tolist()
        
        i = 0
        while i < self.total_samples:
            # Assicurati che current_time_idx sia sempre valido
            current_time_idx = min(i, self.total_samples - 1)
            current_time = self.time_array[current_time_idx]
            glitch_factor_value = float(np.interp(current_time, glitch_factor_time_points, glitch_factor_data))
            glitch_intensity_value = float(np.interp(current_time, glitch_intensity_time_points, glitch_intensity_data))

            # Applica il glitch solo se la probabilità è soddisfatta
            if np.random.rand() < glitch_factor_value:
                glitch_intensity = glitch_intensity_value
                
                # Durata del glitch basata sull'intensità (minimo 1 campione).
                # Tetto massimo alzato da 50ms a 150ms per dare più margine reale allo slider Intensità.
                glitch_duration_samples = int(glitch_intensity * self.sample_rate * 0.15)
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
                        # Scegli il tipo di glitch (pesato da type_probs, uniforme se non specificato)
                        glitch_type = np.random.choice(type_names, p=type_probs)

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
        """Applica un effetto delay dinamico all'audio, processato a piccoli blocchi (non più campione per
        campione). Dato che il tempo di delay minimo possibile (10ms) corrisponde sempre a molti più
        campioni della dimensione del blocco, nessuna lettura ritardata entro un blocco può dipendere da
        dati calcolati nello stesso blocco: questo permette di leggere/scrivere l'intero blocco con
        NumPy vettorizzato invece che con un loop Python per-campione. I parametri (tempo/feedback)
        vengono aggiornati una volta per blocco anziché per ogni singolo campione: una quantizzazione
        finissima (~3ms) che in pratica non si sente, ma è un cambiamento reale rispetto a prima."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_array_processed = audio_array

        delayed_audio = np.copy(audio_array_processed)

        num_channels = delayed_audio.shape[1]
        buffer_len = self.sample_rate
        delay_buffers = [np.zeros(buffer_len, dtype=delayed_audio.dtype) for _ in range(num_channels)]
        write_pos = [0] * num_channels

        # Il delay minimo possibile è 0.01s: il blocco deve restare ben al di sotto di quella soglia
        # in campioni, per garantire che nessuna lettura ritardata ricada nel blocco corrente.
        min_possible_delay_samples = max(1, int(0.01 * self.sample_rate))
        chunk_size = max(1, min(128, min_possible_delay_samples // 2))

        total_samples = len(delayed_audio)

        # I parametri (tempo/feedback) sono letti una sola volta per blocco, non per ogni campione:
        # interpolare l'intero array a risoluzione campione (milioni di punti, via
        # _interp_data_to_audio_length) per poi leggerne solo un valore ogni chunk_size campioni è
        # lavoro sprecato (confermato via profiling: circa un quarto del tempo totale della funzione).
        # Si interpola quindi una sola volta, direttamente nei tempi di inizio-blocco.
        n_chunks = (total_samples + chunk_size - 1) // chunk_size
        chunk_start_indices = np.arange(n_chunks) * chunk_size
        chunk_start_indices = np.minimum(chunk_start_indices, total_samples - 1)
        chunk_times = self.time_array[chunk_start_indices]

        delay_time_time_points = np.linspace(0, self.total_duration_seconds, len(delay_time_data), endpoint=True)
        feedback_time_points = np.linspace(0, self.total_duration_seconds, len(feedback_data), endpoint=True)
        delay_time_at_chunk = np.clip(np.interp(chunk_times, delay_time_time_points, delay_time_data), 0.01, 0.5)
        feedback_at_chunk = np.clip(np.interp(chunk_times, feedback_time_points, feedback_data), 0.0, 0.95)

        # np.arange(seg_len) veniva ricreato ad ogni iterazione del loop (fino a ~100k volte su un
        # video lungo): seg_len è sempre uguale a chunk_size tranne che nell'ultimo blocco, quindi
        # viene calcolato una sola volta e poi affettato.
        arange_chunk = np.arange(chunk_size)

        idx = 0
        chunk_idx = 0
        while idx < total_samples:
            end = min(idx + chunk_size, total_samples)
            seg_len = end - idx

            # np.clip ha un overhead notevole se chiamato su singoli scalari decine di migliaia di
            # volte (macchinari di riduzione generici pensati per array); i valori sono già stati
            # clippati in blocco sopra, quindi qui si legge soltanto il valore già pronto.
            current_delay_time_seconds = float(delay_time_at_chunk[chunk_idx])
            current_feedback_gain = float(feedback_at_chunk[chunk_idx])
            delay_samples = max(1, int(current_delay_time_seconds * self.sample_rate))

            for c in range(num_channels):
                wp = write_pos[c]
                write_positions = (wp + arange_chunk[:seg_len]) % buffer_len
                read_positions = (write_positions - delay_samples) % buffer_len

                delayed_vals = delay_buffers[c][read_positions]
                in_chunk = audio_array_processed[idx:end, c]
                out_chunk = in_chunk + delayed_vals * current_feedback_gain

                delayed_audio[idx:end, c] = out_chunk
                delay_buffers[c][write_positions] = out_chunk
                write_pos[c] = (wp + seg_len) % buffer_len

            idx = end
            chunk_idx += 1

        return delayed_audio.squeeze() if num_channels == 1 else delayed_audio

    def apply_reverb_effect(self, audio_array: np.ndarray, decay_time_data: list, mix_data: list) -> np.ndarray:
        """Applica un semplice effetto di riverbero all'audio, processato a piccoli blocchi (stessa logica
        di apply_delay_effect): i 4 tempi di delay delle linee comb-filter (Schroeder) sono sempre molto
        più lunghi della dimensione del blocco, quindi ogni blocco può essere letto/scritto in modo
        vettorizzato con NumPy invece che campione per campione."""
        if audio_array.ndim == 1:
            audio_array_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_array_processed = audio_array

        reverbed_audio = np.copy(audio_array_processed)

        num_channels = reverbed_audio.shape[1]
        num_delay_lines_per_channel = 4
        buffer_len = self.sample_rate

        # Tempi di delay fissi per un effetto base di riverbero (in campioni)
        # Basati sui valori raccomandati per i delay comb filter (Schroeder)
        delay_times_samples = [
            int(0.0297 * self.sample_rate),
            int(0.0371 * self.sample_rate),
            int(0.0411 * self.sample_rate),
            int(0.0437 * self.sample_rate)
        ]

        delay_lines = [[np.zeros(buffer_len, dtype=reverbed_audio.dtype) for _ in range(num_delay_lines_per_channel)] for _ in range(num_channels)]
        write_pos = [[0] * num_delay_lines_per_channel for _ in range(num_channels)]

        # Il delay più corto tra le 4 linee comb-filter definisce il tetto massimo di sicurezza per il blocco
        min_delay_samples = min(delay_times_samples)
        chunk_size = max(1, min(128, min_delay_samples // 2))

        total_samples = len(reverbed_audio)

        # Stesso fix di apply_delay_effect: i parametri sono letti una volta per blocco, non per
        # campione, quindi si interpola una sola volta nei soli tempi di inizio-blocco invece che
        # sull'intero array a risoluzione campione (qui l'antipattern era ancora più costoso, perché
        # dentro al loop c'era anche un np.arange(seg_len) ricreato per ogni canale x linea di delay
        # x blocco: fino a ~800.000 volte su un video lungo).
        n_chunks = (total_samples + chunk_size - 1) // chunk_size
        chunk_start_indices = np.minimum(np.arange(n_chunks) * chunk_size, total_samples - 1)
        chunk_times = self.time_array[chunk_start_indices]

        decay_time_time_points = np.linspace(0, self.total_duration_seconds, len(decay_time_data), endpoint=True)
        mix_time_points = np.linspace(0, self.total_duration_seconds, len(mix_data), endpoint=True)
        decay_time_at_chunk = np.clip(np.interp(chunk_times, decay_time_time_points, decay_time_data), 0.1, 5.0)
        mix_at_chunk = np.clip(np.interp(chunk_times, mix_time_points, mix_data), 0.0, 1.0)

        # feedback_gain dipende solo da current_decay_time (per blocco) e D (fisso per linea di
        # delay): precalcolabile in blocco per tutti i chunk x tutte le linee in un colpo solo.
        decay_time_col = decay_time_at_chunk[:, None]  # (n_chunks, 1)
        D_row = np.array(delay_times_samples, dtype=np.float64)[None, :]  # (1, 4)
        feedback_gain_at_chunk = np.clip(np.exp(-3 * D_row / (self.sample_rate * decay_time_col)), 0.0, 0.99)  # (n_chunks, 4)

        arange_chunk = np.arange(chunk_size)

        idx = 0
        chunk_idx = 0
        while idx < total_samples:
            end = min(idx + chunk_size, total_samples)
            seg_len = end - idx

            current_mix = float(mix_at_chunk[chunk_idx])

            for c in range(num_channels):
                dry_chunk = audio_array_processed[idx:end, c]
                wet_chunk = np.zeros(seg_len, dtype=reverbed_audio.dtype)

                for dl_idx in range(num_delay_lines_per_channel):
                    D = delay_times_samples[dl_idx]
                    feedback_gain = float(feedback_gain_at_chunk[chunk_idx, dl_idx])

                    wp = write_pos[c][dl_idx]
                    write_positions = (wp + arange_chunk[:seg_len]) % buffer_len
                    read_positions = (write_positions - D) % buffer_len

                    delayed_vals = delay_lines[c][dl_idx][read_positions]
                    wet_chunk = wet_chunk + delayed_vals

                    comb_input = dry_chunk + delayed_vals * feedback_gain
                    delay_lines[c][dl_idx][write_positions] = comb_input
                    write_pos[c][dl_idx] = (wp + seg_len) % buffer_len

                reverbed_audio[idx:end, c] = dry_chunk * (1 - current_mix) + wet_chunk * current_mix * 0.2 # Wet attenuato

            idx = end
            chunk_idx += 1

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

        # Frequenze di taglio per le bande (es. 200 Hz per bassi, 2000 Hz per alti)
        low_cutoff_freq = 200 / nyquist
        high_cutoff_freq = 2000 / nyquist

        # Filtri Butterworth per le bande (questi sono fissi per efficienza)
        # Basse (passa-basso)
        b_low, a_low = butter(2, low_cutoff_freq, btype='low', analog=False)
        # Alte (passa-alto)
        b_high, a_high = butter(2, high_cutoff_freq, btype='high', analog=False)
        # Medie (banda passante, ottenuta sottraendo bassi e alti dal segnale originale)

        # Applica i filtri alle intere tracce per efficienza
        # Questo non è un EQ parametrico frame-per-frame, ma un EQ globale con gain dinamico.
        # Per un EQ dinamico più preciso, avremmo bisogno di implementare filtri che possono cambiare coefficienti al volo (difficile con scipy.signal.lfilter)
        # Alternativa: suddividere l'audio in blocchi e applicare EQ su ogni blocco con i coefficienti attuali.

        # Versione semplificata: applica filtri e poi modula il guadagno delle bande.
        low_band = lfilter(b_low, a_low, eq_audio, axis=0)
        high_band = lfilter(b_high, a_high, eq_audio, axis=0)

        # La banda media è il segnale originale meno le componenti basse e alte (approssimazione)
        mid_band = eq_audio - low_band - high_band # Potrebbe introdurre artefatti per filtri non ideali

        # Applicazione vettorizzata del guadagno dinamico (dB -> lineare per l'intero array in un colpo solo,
        # senza loop Python per-campione/per-canale: stesso risultato, molto più veloce su video lunghi)
        low_gain_linear = 10 ** (low_gain_interp / 20.0)
        mid_gain_linear = 10 ** (mid_gain_interp / 20.0)
        high_gain_linear = 10 ** (high_gain_interp / 20.0)

        if eq_audio.ndim == 2:
            low_gain_linear = low_gain_linear[:, np.newaxis]
            mid_gain_linear = mid_gain_linear[:, np.newaxis]
            high_gain_linear = high_gain_linear[:, np.newaxis]

        eq_audio = (low_band * low_gain_linear +
                    mid_band * mid_gain_linear +
                    high_band * high_gain_linear)

        return eq_audio.squeeze() if eq_audio.shape[1] == 1 else eq_audio

    def apply_elevation_filter(self, audio_array: np.ndarray, elevation_data: list) -> np.ndarray:
        """Simula una sensazione di 'altezza' (sopra/sotto) tramite il timbro, non tramite posizione:
        lo stereo non ha un canale fisico per l'asse verticale, quindi usiamo un trucco psicoacustico
        classico del sound design — un suono più chiaro/brillante viene percepito come "in alto",
        uno più scuro/attutito come "in basso". elevation_data: 0=molto in alto, 1=molto in basso.
        Crossfade vettorizzato tra una versione filtrata passa-basso (scura, "in basso") e il segnale
        pieno (chiaro, "in alto"). Stesso pattern architetturale già usato per l'EQ dinamico."""
        if len(elevation_data) == 0:
            return audio_array

        if audio_array.ndim == 1:
            audio_processed = np.expand_dims(audio_array, axis=1)
        else:
            audio_processed = audio_array

        elevation_interp = self._interp_data_to_audio_length(elevation_data)
        elevation_interp = np.clip(elevation_interp, 0.0, 1.0)

        nyquist = 0.5 * self.sample_rate
        dark_cutoff = 700 / nyquist
        b_dark, a_dark = butter(2, dark_cutoff, btype='low', analog=False)
        dark_signal = lfilter(b_dark, a_dark, audio_processed, axis=0)

        # elevation basso (in alto nel frame) -> peso alto sul segnale pieno/chiaro
        # elevation alto (in basso nel frame) -> peso alto sul segnale scuro/filtrato
        bright_weight = 1.0 - elevation_interp
        dark_weight = elevation_interp

        if audio_processed.ndim == 2:
            bright_weight = bright_weight[:, np.newaxis]
            dark_weight = dark_weight[:, np.newaxis]

        result = audio_processed * bright_weight + dark_signal * dark_weight

        return result.squeeze() if result.shape[1] == 1 else result

    def apply_stereo_panning(self, audio_array: np.ndarray, pan_data: list) -> np.ndarray:
        """Applica un panning stereo dinamico a un segnale mono, basato su pan_data
        (valori 0=sinistra, 0.5=centro, 1=destra, tipicamente il centro di massa orizzontale del video).
        Usa una legge a potenza costante (constant-power pan law) per un movimento naturale nello
        spazio stereo. Completamente vettorizzato: nessun loop per-campione.
        Se l'audio è già multi-canale, viene restituito invariato (non sovrascriviamo un mix già stereo)."""
        if audio_array.ndim != 1:
            return audio_array

        pan_interp = self._interp_data_to_audio_length(pan_data)
        pan_interp = np.clip(pan_interp, 0.0, 1.0)

        angle = pan_interp * (np.pi / 2.0)
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        left_channel = audio_array * left_gain
        right_channel = audio_array * right_gain

        return np.stack([left_channel, right_channel], axis=1)


@st.cache_resource(show_spinner=False)
def get_audio_generator(sample_rate: int, total_duration_seconds: float) -> AudioGenerator:
    """Crea (e riusa tra un rerun e l'altro) l'AudioGenerator per una data combinazione
    sample_rate/durata. Costruire self.time_array nel costruttore — un array NumPy da milioni
    di campioni per un video di qualche minuto — costa qualche centinaio di millisecondi;
    senza questa cache verrebbe rifatto da capo ad OGNI interazione con l'interfaccia (ogni
    checkbox o slider toccato fa ripartire l'intero script Streamlit), non solo quando serve
    davvero. AudioGenerator non modifica mai il proprio stato dopo la costruzione (nessun
    self.xxx = ... fuori da __init__), quindi riusare la stessa istanza tra i rerun è sicuro."""
    return AudioGenerator(sample_rate=sample_rate, total_duration_seconds=total_duration_seconds)


def main():
    st.set_page_config(layout="wide", page_title="VideoSound Gen. by Loop507", page_icon="🎵")

    # Modifica 1: Titolo con "by Loop507" più piccolo
    st.markdown("# VideoSound Gen. <small>by Loop507</small>", unsafe_allow_html=True)
    st.markdown("Crea colonne sonore uniche dai tuoi video, trasformando i dati visivi in paesaggi sonori dinamici.")

    st.sidebar.header("Sorgente")
    input_mode = st.sidebar.radio(
        "Modalità", ["🎬 Da Video", "🎲 Generativa (senza video)"], key='input_mode',
        help="'Generativa' usa VideoSound-Gen come sintetizzatore standalone: le stesse tecniche di "
             "sintesi/effetti/preset, ma pilotate da curve procedurali (LFO + rumore smussato) invece "
             "che dall'analisi di un video — comodo per creare un brano senza dover caricare nulla."
    )
    generative_mode = (input_mode == "🎲 Generativa (senza video)")

    if generative_mode:
        uploaded_file = None
        st.sidebar.subheader("Impostazioni Generativa")
        generative_duration = st.sidebar.slider("Durata Brano (sec)", 5, MAX_DURATION, 60, key='generative_duration')
        generative_track_name = st.sidebar.text_input("Nome Traccia", value="traccia_generativa", key='generative_track_name')
        generative_seed_enabled = st.sidebar.checkbox(
            "Seed fisso (riproducibile)", value=False, key='generative_seed_on',
            help="Se disattivo, ogni rigenerazione produce curve leggermente diverse. Se attivo, "
                 "la stessa curva si ripete finché non cambi il seed."
        )
        generative_seed = st.sidebar.number_input("Seed", 0, 999999, 42, key='generative_seed') if generative_seed_enabled else None
    else:
        st.sidebar.header("Carica Video")
        uploaded_file = st.sidebar.file_uploader("Scegli un file video (MP4, MOV, AVI, ecc.)", type=["mp4", "mov", "avi", "mkv"])

    # Variabili per memorizzare i parametri scelti dall'utente per la descrizione finale
    params = {}

    # Inizializza audio_output_path e base_name_output a None per evitare UnboundLocalError
    # Saranno sovrascritti se il pulsante viene cliccato e la generazione audio avviene.
    audio_output_path = None
    base_name_output = None

    # Inizializza session_state per i download (persistono tra i rerun)
    if 'video_bytes' not in st.session_state:
        st.session_state['video_bytes'] = None
    if 'stems_zip_bytes' not in st.session_state:
        st.session_state['stems_zip_bytes'] = None
    if 'generative_audio_bytes' not in st.session_state:
        st.session_state['generative_audio_bytes'] = None
    if 'video_filename' not in st.session_state:
        st.session_state['video_filename'] = None
    if 'report_text' not in st.session_state:
        st.session_state['report_text'] = None
    if 'report_filename' not in st.session_state:
        st.session_state['report_filename'] = None

    # ID univoco per sessione: usato in tutti i nomi dei file temporanei per evitare
    # collisioni quando più utenti usano l'app contemporaneamente sulla stessa istanza
    # (es. Streamlit Cloud, dove i processi condividono la working directory).
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = uuid.uuid4().hex[:8]
    session_id = st.session_state['session_id']

    if uploaded_file is not None or generative_mode:
        video_input_path = None  # resta None in modalità generativa: nessun file video reale

        if generative_mode:
            luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data, vertical_mass_center_data, edge_density_data, color_variation_data, duration_seconds, fps = generate_synthetic_feature_curves(
                float(generative_duration), fps=30.0, seed=generative_seed
            )
            base_name_output = generative_track_name.strip() or "traccia_generativa"

        else:
            if not validate_video_file(uploaded_file):
                # Se la validazione fallisce, pulisci il file caricato e termina
                invalid_path = f"temp_input_{session_id}_{uploaded_file.name}"
                if os.path.exists(invalid_path):
                    os.remove(invalid_path)
                return

            st.sidebar.success("✅ Video caricato con successo!")

            # Pulisci eventuali file temporanei rimasti da un upload precedente in questa
            # stessa sessione (es. utente carica un video, non completa la generazione,
            # poi ne carica un altro: senza questa pulizia i vecchi file si accumulerebbero).
            cleanup_session_temp_files(session_id)

            # Salva il file temporaneamente, con nome univoco per sessione
            video_input_path = f"temp_input_{session_id}_{uploaded_file.name}"
            with open(video_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Hash del contenuto: usato come chiave di cache insieme al path (vedi il commento su
            # analyze_video_frames_cached) per evitare che un video diverso con lo stesso nome file
            # riusi per errore l'analisi di un video precedente nella stessa sessione.
            file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()

            luminosity_data, detail_data, movement_data, variation_movement_data, horizontal_mass_center_data, vertical_mass_center_data, edge_density_data, color_variation_data, duration_seconds, fps = analyze_video_frames_cached(video_input_path, file_hash)

            if duration_seconds == 0.0: # Se l'analisi fallisce o video troppo corto/lungo
                os.remove(video_input_path)
                return

            # base_name_output è ora definito qui, prima dell'uso condizionale
            base_name_output = os.path.splitext(uploaded_file.name)[0]

        st.subheader("Generazione Audio")

        # ── BOX PRESET PER TIPO DI VIDEO ────────────────────────────────────
        st.markdown("#### 🎨 Preset per tipo di video")
        st.caption("Scegli un punto di partenza in base al tipo di footage, poi ritocca liberamente i parametri nei tab sotto.")
        preset_options = [PRESET_MANUALE] + list(VIDEO_PRESETS.keys())
        selected_preset = st.selectbox("Preset", preset_options, key='preset_choice')

        if selected_preset != PRESET_MANUALE and st.session_state.get('_last_applied_preset') != selected_preset:
            # Azzera tutti i toggle dei layer prima di applicare il preset specifico: altrimenti
            # un layer abilitato da un preset precedente (o acceso manualmente) resterebbe attivo
            # anche scegliendo un nuovo preset che non lo nomina esplicitamente nel suo dizionario.
            all_layer_toggles = [
                'subtractive_on', 'sub_filter_on', 'sub_band_limited', 'fm_on', 'epiano_on',
                'granular_on', 'pluck_on', 'noise_on', 'glitch_on', 'delay_on', 'reverb_on',
                'eq_on', 'panning_on', 'elevation_on', 'melody_on',
            ]
            for toggle_key in all_layer_toggles:
                st.session_state[toggle_key] = False
            for preset_key, preset_value in VIDEO_PRESETS[selected_preset].items():
                st.session_state[preset_key] = preset_value
            st.session_state['_last_applied_preset'] = selected_preset
            st.rerun()
        elif selected_preset == PRESET_MANUALE:
            st.session_state['_last_applied_preset'] = None

        # Inizializza AudioGenerator
        audio_generator = get_audio_generator(AUDIO_SAMPLE_RATE, duration_seconds)

        # ── Melodia globale (quantizzazione a scala) ──────────────────────────────
        # Un unico controllo condiviso da tutti i layer con un'intonazione (sottrattiva, FM,
        # granulare, e-piano, corde) invece di un controllo per-layer: se ogni layer avesse la
        # propria scala/tonica indipendente, potrebbero suonare in chiavi diverse simultaneamente.
        # Rumore e Glitch non compaiono qui: sono broadband/senza un'intonazione riconoscibile
        # (il rumore è per definizione energia distribuita su tutte le frequenze; il glitch lavora
        # su ripetizione/rumore/inversione del segnale, non su un pitch) — non c'è nulla da
        # agganciare a una nota, quindi "quantizzarli" non avrebbe alcun effetto reale.
        with st.expander("🎵 Melodia (quantizzazione a scala)", expanded=False):
            st.caption(
                "Di default l'intonazione di ogni layer è continua, guidata liberamente dal video "
                "(come un glissando). Abilitando la quantizzazione, ogni nota/pizzicata/oscillatore "
                "si aggancia alla nota più vicina della scala scelta: il video continua a decidere "
                "quale nota suona, ma il risultato è una melodia riconoscibile invece di "
                "un'intonazione libera/microtonale. Si applica a tutti i layer con un'intonazione "
                "(Sottrattiva, FM, Granulare, E-Piano, Corde) usando la stessa tonica, così restano "
                "sempre in accordo tra loro invece che ognuno per conto suo."
            )
            use_melody_quantization = st.checkbox("Abilita Melodia", value=False, key='melody_on')
            params['melody_enabled'] = use_melody_quantization
            if use_melody_quantization:
                melody_scale = st.selectbox("Scala Musicale", list(MUSICAL_SCALES.keys()), key='melody_scale')
                melody_root_note = st.slider(
                    "Nota Fondamentale (Hz)", 55.0, 440.0, 220.0, step=1.0, key='melody_root_note',
                    help="La 'tonica' condivisa da tutti i layer: 220Hz = La3, 110Hz = La2, 261.6Hz = Do4 (centrale)."
                )
            else:
                melody_scale = SCALE_NONE
                melody_root_note = 220.0
            params['melody_scale'] = melody_scale
            params['melody_root_note'] = melody_root_note

        def apply_melody(freqs: list) -> list:
            """Applica la quantizzazione a scala globale, se abilitata, altrimenti restituisce
            le frequenze invariate (intonazione libera)."""
            if not use_melody_quantization:
                return freqs
            return quantize_to_scale(freqs, melody_root_note, melody_scale)

        # Scheda per i parametri audio
        tab_sub, tab_fm, tab_epiano, tab_gran, tab_pluck, tab_noise, tab_fx, tab_eq, tab_pan = st.tabs([
            "Sintesi Sottrattiva", "Sintesi FM", "E-Piano (FM)", "Sintesi Granulare", "Corde (Karplus-Strong)", "Rumore", "Effetti Audio", "Equalizzatore", "Panning & Altezza"
        ])

        # Layer 1: Sintesi Sottrattiva (Basato su Luminosità e Dettaglio)
        with tab_sub:
            st.markdown("### Layer: Sintesi Sottrattiva")
            use_subtractive = st.checkbox("Abilita Sintesi Sottrattiva", value=True, key='subtractive_on')
            params['subtractive_enabled'] = use_subtractive
            if use_subtractive:
                sub_layer_gain = st.slider("🎚️ Volume Layer Sottrattivo", 0.0, 2.0, 1.0, step=0.05, key='sub_gain',
                                            help="Bilancia questo layer rispetto agli altri attivi contemporaneamente.")
                params['sub_layer_gain'] = sub_layer_gain
            else:
                sub_layer_gain = 0.0
            if use_subtractive:
                sub_freq_source = st.selectbox("Sorgente Frequenza (Hz)", ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"], key='sub_freq_src')
                sub_amp_source = st.selectbox("Sorgente Ampiezza (0-1)", ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"], key='sub_amp_src')
                sub_waveform_type = st.selectbox("Tipo di Forma d'Onda", ["sine", "square", "sawtooth"], key='sub_waveform_type')

                if sub_waveform_type in ("square", "sawtooth"):
                    sub_band_limited = st.checkbox(
                        "Band-limited (anti-aliasing PolyBLEP)", value=False, key='sub_band_limited',
                        help="Di default square/sawtooth sono generate 'al naturale' e aliasano alle "
                             "frequenze medio-alte: è l'artefatto lo-fi/glitch tipico ricercato in molta "
                             "musica elettronica. Attiva questa opzione per un suono più pulito in stile "
                             "synth analogico, senza aliasing."
                    )
                else:
                    sub_band_limited = False
                params['sub_band_limited'] = sub_band_limited

                
                sub_freq_min = st.slider("Frequenza Minima (Hz)", 20, 1000, 100, key='sub_freq_min')
                sub_freq_max = st.slider("Frequenza Massima (Hz)", 20, 1000, 800, key='sub_freq_max')
                sub_amp_min = st.slider("Ampiezza Minima", 0.0, 1.0, 0.1, step=0.01, key='sub_amp_min')
                sub_amp_max = st.slider("Ampiezza Massima", 0.0, 1.0, 0.5, step=0.01, key='sub_amp_max')

                params['sub_freq_source'] = sub_freq_source
                params['sub_amp_source'] = sub_amp_source
                params['sub_waveform_type'] = sub_waveform_type
                params['sub_freq_range'] = (sub_freq_min, sub_freq_max)
                params['sub_amp_range'] = (sub_amp_min, sub_amp_max)

                sub_freq_data_raw = []
                if sub_freq_source == "Luminosità": sub_freq_data_raw = luminosity_data
                elif sub_freq_source == "Dettaglio": sub_freq_data_raw = detail_data
                elif sub_freq_source == "Movimento": sub_freq_data_raw = movement_data
                elif sub_freq_source == "Densità Contorni": sub_freq_data_raw = edge_density_data
                elif sub_freq_source == "Variazione Colore": sub_freq_data_raw = color_variation_data

                sub_amp_data_raw = []
                if sub_amp_source == "Luminosità": sub_amp_data_raw = luminosity_data
                elif sub_amp_source == "Dettaglio": sub_amp_data_raw = detail_data
                elif sub_amp_source == "Movimento": sub_amp_data_raw = movement_data
                elif sub_amp_source == "Densità Contorni": sub_amp_data_raw = edge_density_data
                elif sub_amp_source == "Variazione Colore": sub_amp_data_raw = color_variation_data
                
                # Normalizza e scala i dati delle sorgenti
                sub_freq_scaled = scale_frequency_exponential(sub_freq_data_raw, sub_freq_min, sub_freq_max)
                sub_freq_scaled = apply_melody(sub_freq_scaled)
                sub_amp_scaled = np.interp(sub_amp_data_raw, (min(sub_amp_data_raw) if sub_amp_data_raw else 0, max(sub_amp_data_raw) if sub_amp_data_raw else 1), (sub_amp_min, sub_amp_max)).tolist()

                st.markdown("##### Filtro Sottrattivo (vero)")
                sub_filter_enabled = st.checkbox(
                    "Applica filtro passa-basso risonante", value=False, key='sub_filter_on',
                    help="'Sintesi sottrattiva' significa scolpire le armoniche con un filtro, non solo "
                         "modulare ampiezza/frequenza dell'onda: questo aggiunge il filtro vero e proprio "
                         "(risonante, con Q) sopra la forma d'onda già generata. Funziona meglio con "
                         "square/sawtooth, che sono ricche di armoniche da tagliare."
                )
                params['sub_filter_enabled'] = sub_filter_enabled
                if sub_filter_enabled:
                    sub_cutoff_source = st.selectbox("Sorgente Cutoff (Hz)", ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"], key='sub_cutoff_src')
                    sub_cutoff_min = st.slider("Cutoff Minimo (Hz)", 50, 5000, 200, key='sub_cutoff_min')
                    sub_cutoff_max = st.slider("Cutoff Massimo (Hz)", 50, 8000, 3000, key='sub_cutoff_max')
                    sub_resonance = st.slider(
                        "Risonanza (Q)", 0.5, 8.0, 1.5, step=0.1, key='sub_resonance',
                        help="Q alto = picco più marcato intorno al cutoff (carattere più 'synth'/aggressivo)."
                    )
                    params['sub_cutoff_source'] = sub_cutoff_source
                    params['sub_cutoff_range'] = (sub_cutoff_min, sub_cutoff_max)
                    params['sub_resonance'] = sub_resonance

                    sub_cutoff_data_raw = []
                    if sub_cutoff_source == "Luminosità": sub_cutoff_data_raw = luminosity_data
                    elif sub_cutoff_source == "Dettaglio": sub_cutoff_data_raw = detail_data
                    elif sub_cutoff_source == "Movimento": sub_cutoff_data_raw = movement_data
                    elif sub_cutoff_source == "Densità Contorni": sub_cutoff_data_raw = edge_density_data
                    elif sub_cutoff_source == "Variazione Colore": sub_cutoff_data_raw = color_variation_data

                    sub_cutoff_scaled = scale_frequency_exponential(sub_cutoff_data_raw, sub_cutoff_min, sub_cutoff_max)
                    sub_resonance_scaled = [sub_resonance] * max(len(sub_cutoff_scaled), 1)
                else:
                    sub_cutoff_scaled = []
                    sub_resonance_scaled = []
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                sub_freq_scaled = []
                sub_amp_scaled = []
                sub_band_limited = False
                params['sub_band_limited'] = False
                sub_filter_enabled = False
                params['sub_filter_enabled'] = False
                sub_cutoff_scaled = []
                sub_resonance_scaled = []

        # Layer 2: Sintesi FM (Basato su Variazione Movimento e Centro di Massa Orizzontale)
        with tab_fm:
            st.markdown("### Layer: Sintesi FM")
            use_fm = st.checkbox("Abilita Sintesi FM", value=True, key='fm_on')
            params['fm_enabled'] = use_fm
            if use_fm:
                fm_layer_gain = st.slider("🎚️ Volume Layer FM", 0.0, 2.0, 1.0, step=0.05, key='fm_gain',
                                           help="Bilancia questo layer rispetto agli altri attivi contemporaneamente.")
                params['fm_layer_gain'] = fm_layer_gain

                fm_source_options = ["Luminosità", "Dettaglio", "Movimento", "Variazione Movimento", "Densità Contorni", "Variazione Colore"]
                fm_carrier_source = st.selectbox("Sorgente Frequenza Portante (Hz)", fm_source_options, key='fm_carr_src')
                fm_mod_source = st.selectbox("Sorgente Frequenza Modulatore (Hz)", fm_source_options, key='fm_mod_src')
                fm_mod_idx_source = st.selectbox("Sorgente Indice Modulazione", fm_source_options, key='fm_idx_src')
                fm_amp_source = st.selectbox("Sorgente Ampiezza (0-1)", fm_source_options, key='fm_amp_src')

                fm_carrier_min = st.slider("Portante Minima (Hz)", 50, 2000, 200, key='fm_carr_min')
                fm_carrier_max = st.slider("Portante Massima (Hz)", 50, 2000, 1500, key='fm_carr_max')
                fm_mod_min = st.slider("Modulatore Minimo (Hz)", 10, 500, 50, key='fm_mod_min')
                fm_mod_max = st.slider("Modulatore Massimo (Hz)", 10, 500, 250, key='fm_mod_max')
                fm_mod_idx_min = st.slider("Indice Modulazione Minimo", 0.0, 10.0, 0.5, step=0.1, key='fm_idx_min')
                fm_mod_idx_max = st.slider("Indice Modulazione Massimo", 0.0, 10.0, 5.0, step=0.1, key='fm_idx_max')
                fm_amp_min = st.slider("Ampiezza FM Minima", 0.0, 1.0, 0.05, step=0.01, key='fm_amp_min')
                fm_amp_max = st.slider("Ampiezza FM Massima", 0.0, 1.0, 0.3, step=0.01, key='fm_amp_max')

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
                elif fm_carrier_source == "Densità Contorni": fm_carrier_data_raw = edge_density_data
                elif fm_carrier_source == "Variazione Colore": fm_carrier_data_raw = color_variation_data

                fm_mod_data_raw = []
                if fm_mod_source == "Luminosità": fm_mod_data_raw = luminosity_data
                elif fm_mod_source == "Dettaglio": fm_mod_data_raw = detail_data
                elif fm_mod_source == "Movimento": fm_mod_data_raw = movement_data
                elif fm_mod_source == "Variazione Movimento": fm_mod_data_raw = variation_movement_data
                elif fm_mod_source == "Densità Contorni": fm_mod_data_raw = edge_density_data
                elif fm_mod_source == "Variazione Colore": fm_mod_data_raw = color_variation_data

                fm_mod_idx_data_raw = []
                if fm_mod_idx_source == "Luminosità": fm_mod_idx_data_raw = luminosity_data
                elif fm_mod_idx_source == "Dettaglio": fm_mod_idx_data_raw = detail_data
                elif fm_mod_idx_source == "Movimento": fm_mod_idx_data_raw = movement_data
                elif fm_mod_idx_source == "Variazione Movimento": fm_mod_idx_data_raw = variation_movement_data
                elif fm_mod_idx_source == "Densità Contorni": fm_mod_idx_data_raw = edge_density_data
                elif fm_mod_idx_source == "Variazione Colore": fm_mod_idx_data_raw = color_variation_data

                fm_amp_data_raw = []
                if fm_amp_source == "Luminosità": fm_amp_data_raw = luminosity_data
                elif fm_amp_source == "Dettaglio": fm_amp_data_raw = detail_data
                elif fm_amp_source == "Movimento": fm_amp_data_raw = movement_data
                elif fm_amp_source == "Variazione Movimento": fm_amp_data_raw = variation_movement_data
                elif fm_amp_source == "Densità Contorni": fm_amp_data_raw = edge_density_data
                elif fm_amp_source == "Variazione Colore": fm_amp_data_raw = color_variation_data


                fm_carrier_scaled = scale_frequency_exponential(fm_carrier_data_raw, fm_carrier_min, fm_carrier_max)
                fm_carrier_scaled = apply_melody(fm_carrier_scaled)
                fm_mod_scaled = scale_frequency_exponential(fm_mod_data_raw, fm_mod_min, fm_mod_max)
                fm_mod_idx_scaled = np.interp(fm_mod_idx_data_raw, (min(fm_mod_idx_data_raw) if fm_mod_idx_data_raw else 0, max(fm_mod_idx_data_raw) if fm_mod_idx_data_raw else 1), (fm_mod_idx_min, fm_mod_idx_max)).tolist()
                fm_amp_scaled = np.interp(fm_amp_data_raw, (min(fm_amp_data_raw) if fm_amp_data_raw else 0, max(fm_amp_data_raw) if fm_amp_data_raw else 1), (fm_amp_min, fm_amp_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                fm_carrier_scaled = []
                fm_mod_scaled = []
                fm_mod_idx_scaled = []
                fm_amp_scaled = []
                fm_layer_gain = 0.0

        # Layer 2b: E-Piano FM (note discrete, stile DX7/Rhodes)
        with tab_epiano:
            st.markdown("### Layer: E-Piano (FM a note)")
            st.caption(
                "A differenza della Sintesi FM continua qui sopra, questo layer suona NOTE "
                "discrete con inviluppo: attacco rapido, decadimento esponenziale, e un indice "
                "di modulazione che si smorza più in fretta dell'ampiezza (il timbro si "
                "'ammorbidisce' mentre la nota risuona) — il meccanismo classico degli e-piano "
                "FM digitali (DX7 EPiano1, Rhodes)."
            )
            use_epiano = st.checkbox("Abilita Layer E-Piano", value=False, key='epiano_on')
            params['epiano_enabled'] = use_epiano
            if use_epiano:
                epiano_layer_gain = st.slider("🎚️ Volume Layer E-Piano", 0.0, 2.0, 1.0, step=0.05, key='epiano_gain')
                params['epiano_layer_gain'] = epiano_layer_gain

                epiano_source_options = ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"]
                epiano_density_source = st.selectbox("Sorgente Densità Note", epiano_source_options, key='epiano_dens_src')
                epiano_pitch_source = st.selectbox("Sorgente Intonazione", epiano_source_options, key='epiano_pitch_src', index=2)
                epiano_brightness_source = st.selectbox("Sorgente Brillantezza (indice FM)", epiano_source_options, key='epiano_bright_src', index=1)
                epiano_amp_source = st.selectbox("Sorgente Ampiezza", epiano_source_options, key='epiano_amp_src')

                epiano_density_min = st.slider("Densità Minima (note/finestra)", 0, 5, 0, key='epiano_dens_min')
                epiano_density_max = st.slider("Densità Massima (note/finestra)", 0, 5, 2, key='epiano_dens_max')
                epiano_pitch_min = st.slider("Intonazione Minima (Hz)", 40, 800, 130, key='epiano_pitch_min')
                epiano_pitch_max = st.slider("Intonazione Massima (Hz)", 40, 1500, 500, key='epiano_pitch_max')
                epiano_amp_min = st.slider("Ampiezza Minima", 0.0, 1.0, 0.3, step=0.01, key='epiano_amp_min')
                epiano_amp_max = st.slider("Ampiezza Massima", 0.0, 1.0, 1.0, step=0.01, key='epiano_amp_max')
                epiano_note_duration = st.slider(
                    "Durata Massima Nota (sec)", 0.3, 2.5, 1.2, step=0.1, key='epiano_note_dur',
                    help="Quanto a lungo può risuonare la nota prima di essere tagliata."
                )
                epiano_mod_ratio = st.select_slider(
                    "Rapporto Modulatore/Portante", options=[0.5, 1.0, 1.4, 2.0, 3.5, 7.0, 14.0], value=1.0, key='epiano_mod_ratio',
                    help="Rapporti classici della sintesi FM: 1.0 e 1.4 danno timbri 'e-piano' morbidi, "
                         "3.5/7.0 timbri più metallici/campanellati, 14.0 è il rapporto usato nella terza "
                         "coppia operatore del patch 'E.Piano 1' originale del DX7 per l'attacco a campana."
                )

                params['epiano_density_source'] = epiano_density_source
                params['epiano_pitch_source'] = epiano_pitch_source
                params['epiano_brightness_source'] = epiano_brightness_source
                params['epiano_amp_source'] = epiano_amp_source
                params['epiano_density_range'] = (epiano_density_min, epiano_density_max)
                params['epiano_pitch_range'] = (epiano_pitch_min, epiano_pitch_max)
                params['epiano_amp_range'] = (epiano_amp_min, epiano_amp_max)
                params['epiano_note_duration'] = epiano_note_duration
                params['epiano_mod_ratio'] = epiano_mod_ratio

                def _epiano_source_data(name):
                    if name == "Luminosità": return luminosity_data
                    if name == "Dettaglio": return detail_data
                    if name == "Movimento": return movement_data
                    if name == "Densità Contorni": return edge_density_data
                    if name == "Variazione Colore": return color_variation_data
                    return []

                epiano_density_data_raw = _epiano_source_data(epiano_density_source)
                epiano_pitch_data_raw = _epiano_source_data(epiano_pitch_source)
                epiano_brightness_data_raw = _epiano_source_data(epiano_brightness_source)
                epiano_amp_data_raw = _epiano_source_data(epiano_amp_source)

                epiano_density_scaled = np.interp(epiano_density_data_raw, (min(epiano_density_data_raw) if epiano_density_data_raw else 0, max(epiano_density_data_raw) if epiano_density_data_raw else 1), (epiano_density_min, epiano_density_max)).tolist()
                epiano_pitch_scaled = scale_frequency_exponential(epiano_pitch_data_raw, epiano_pitch_min, epiano_pitch_max)
                epiano_pitch_scaled = apply_melody(epiano_pitch_scaled)
                # Brillantezza normalizzata in [0,1] indipendentemente dal range della sorgente scelta
                epiano_brightness_scaled = np.interp(epiano_brightness_data_raw, (min(epiano_brightness_data_raw) if epiano_brightness_data_raw else 0, max(epiano_brightness_data_raw) if epiano_brightness_data_raw else 1), (0.0, 1.0)).tolist()
                epiano_amp_scaled = np.interp(epiano_amp_data_raw, (min(epiano_amp_data_raw) if epiano_amp_data_raw else 0, max(epiano_amp_data_raw) if epiano_amp_data_raw else 1), (epiano_amp_min, epiano_amp_max)).tolist()
            else:
                epiano_layer_gain = 0.0
                epiano_density_scaled = []
                epiano_pitch_scaled = []
                epiano_brightness_scaled = []
                epiano_amp_scaled = []
                epiano_note_duration = 1.2
                epiano_mod_ratio = 1.0
                params['epiano_note_duration'] = epiano_note_duration
                params['epiano_mod_ratio'] = epiano_mod_ratio

        # Layer 3: Sintesi Granulare (Basato su Dettaglio e Movimento)
        with tab_gran:
            st.markdown("### Layer: Sintesi Granulare")
            use_granular = st.checkbox("Abilita Sintesi Granulare", value=True, key='granular_on')
            params['granular_enabled'] = use_granular
            if use_granular:
                gran_layer_gain = st.slider("🎚️ Volume Layer Granulare", 0.0, 2.0, 1.0, step=0.05, key='gran_gain',
                                             help="Bilancia questo layer rispetto agli altri attivi contemporaneamente.")
                params['gran_layer_gain'] = gran_layer_gain

                gran_source_options = ["Dettaglio", "Movimento", "Variazione Movimento", "Densità Contorni", "Variazione Colore"]
                gran_density_source = st.selectbox("Sorgente Densità Grani", gran_source_options, key='gran_dens_src')
                gran_duration_source = st.selectbox("Sorgente Durata Grani (sec)", gran_source_options, key='gran_dur_src')
                gran_amp_source = st.selectbox("Sorgente Ampiezza Grani (0-1)", gran_source_options, key='gran_amp_src')
                gran_pitch_source = st.selectbox("Sorgente Intonazione Grani (Hz)", ["Luminosità"] + gran_source_options, key='gran_pitch_src',
                                                  help="Prima era sempre casuale: ora anche l'intonazione di ogni grano segue il video.")
                
                gran_density_min = st.slider("Densità Minima Grani", 0, 10, 1, key='gran_dens_min')
                gran_density_max = st.slider("Densità Massima Grani", 0, 10, 5, key='gran_dens_max')
                gran_duration_min = st.slider("Durata Minima Grani (sec)", 0.01, 0.1, 0.02, step=0.005, key='gran_dur_min')
                gran_duration_max = st.slider("Durata Massima Grani (sec)", 0.01, 0.1, 0.05, step=0.005, key='gran_dur_max')
                gran_amp_min = st.slider("Ampiezza Grani Minima", 0.0, 1.0, 0.01, step=0.01, key='gran_amp_min')
                gran_amp_max = st.slider("Ampiezza Grani Massima", 0.0, 1.0, 0.1, step=0.01, key='gran_amp_max')
                gran_pitch_min = st.slider("Intonazione Minima Grani (Hz)", 50, 2000, 200, key='gran_pitch_min')
                gran_pitch_max = st.slider("Intonazione Massima Grani (Hz)", 50, 2000, 1000, key='gran_pitch_max')

                params['gran_density_source'] = gran_density_source
                params['gran_duration_source'] = gran_duration_source
                params['gran_amp_source'] = gran_amp_source
                params['gran_pitch_source'] = gran_pitch_source
                params['gran_density_range'] = (gran_density_min, gran_density_max)
                params['gran_duration_range'] = (gran_duration_min, gran_duration_max)
                params['gran_amp_range'] = (gran_amp_min, gran_amp_max)
                params['gran_pitch_range'] = (gran_pitch_min, gran_pitch_max)

                gran_density_data_raw = []
                if gran_density_source == "Dettaglio": gran_density_data_raw = detail_data
                elif gran_density_source == "Movimento": gran_density_data_raw = movement_data
                elif gran_density_source == "Variazione Movimento": gran_density_data_raw = variation_movement_data
                elif gran_density_source == "Densità Contorni": gran_density_data_raw = edge_density_data
                elif gran_density_source == "Variazione Colore": gran_density_data_raw = color_variation_data

                gran_duration_data_raw = []
                if gran_duration_source == "Dettaglio": gran_duration_data_raw = detail_data
                elif gran_duration_source == "Movimento": gran_duration_data_raw = movement_data
                elif gran_duration_source == "Variazione Movimento": gran_duration_data_raw = variation_movement_data
                elif gran_duration_source == "Densità Contorni": gran_duration_data_raw = edge_density_data
                elif gran_duration_source == "Variazione Colore": gran_duration_data_raw = color_variation_data

                gran_amp_data_raw = []
                if gran_amp_source == "Dettaglio": gran_amp_data_raw = detail_data
                elif gran_amp_source == "Movimento": gran_amp_data_raw = movement_data
                elif gran_amp_source == "Variazione Movimento": gran_amp_data_raw = variation_movement_data
                elif gran_amp_source == "Densità Contorni": gran_amp_data_raw = edge_density_data
                elif gran_amp_source == "Variazione Colore": gran_amp_data_raw = color_variation_data

                gran_pitch_data_raw = []
                if gran_pitch_source == "Luminosità": gran_pitch_data_raw = luminosity_data
                elif gran_pitch_source == "Dettaglio": gran_pitch_data_raw = detail_data
                elif gran_pitch_source == "Movimento": gran_pitch_data_raw = movement_data
                elif gran_pitch_source == "Variazione Movimento": gran_pitch_data_raw = variation_movement_data
                elif gran_pitch_source == "Densità Contorni": gran_pitch_data_raw = edge_density_data
                elif gran_pitch_source == "Variazione Colore": gran_pitch_data_raw = color_variation_data

                gran_density_scaled = np.interp(gran_density_data_raw, (min(gran_density_data_raw) if gran_density_data_raw else 0, max(gran_density_data_raw) if gran_density_data_raw else 1), (gran_density_min, gran_density_max)).tolist()
                gran_duration_scaled = np.interp(gran_duration_data_raw, (min(gran_duration_data_raw) if gran_duration_data_raw else 0, max(gran_duration_data_raw) if gran_duration_data_raw else 1), (gran_duration_min, gran_duration_max)).tolist()
                gran_amp_scaled = np.interp(gran_amp_data_raw, (min(gran_amp_data_raw) if gran_amp_data_raw else 0, max(gran_amp_data_raw) if gran_amp_data_raw else 1), (gran_amp_min, gran_amp_max)).tolist()
                gran_pitch_scaled = scale_frequency_exponential(gran_pitch_data_raw, gran_pitch_min, gran_pitch_max)
                gran_pitch_scaled = apply_melody(gran_pitch_scaled)
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                gran_density_scaled = []
                gran_duration_scaled = []
                gran_amp_scaled = []
                gran_pitch_scaled = []
                gran_layer_gain = 0.0


        # Layer 3b: Corde (Karplus-Strong, modellazione fisica)
        with tab_pluck:
            st.markdown("### Layer: Corde (Karplus-Strong)")
            st.caption(
                "Modellazione fisica: simula una corda pizzicata (rumore che circola in una linea "
                "di ritardo accordata, con smorzamento) invece di un'onda o un grano. Carattere "
                "percussivo/organico, diverso dagli altri layer."
            )
            use_pluck = st.checkbox("Abilita Layer Corde", value=False, key='pluck_on')
            params['pluck_enabled'] = use_pluck
            if use_pluck:
                pluck_layer_gain = st.slider("🎚️ Volume Layer Corde", 0.0, 2.0, 1.0, step=0.05, key='pluck_gain')
                params['pluck_layer_gain'] = pluck_layer_gain

                pluck_source_options = ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"]
                pluck_density_source = st.selectbox("Sorgente Densità Pizzicate", pluck_source_options, key='pluck_dens_src')
                pluck_pitch_source = st.selectbox("Sorgente Intonazione", pluck_source_options, key='pluck_pitch_src', index=2)
                pluck_damping_source = st.selectbox("Sorgente Smorzamento", pluck_source_options, key='pluck_damp_src', index=1)
                pluck_amp_source = st.selectbox("Sorgente Ampiezza", pluck_source_options, key='pluck_amp_src')

                # Densità volutamente limitata (max 5, non 10 come il layer granulare): pizzicate
                # più fitte di ~150ms perderebbero il carattere "corda" per fondersi in una texture
                # continua (che è già il ruolo della sintesi granulare), quindi non aggiungerebbero
                # nulla di espressivo — solo tempo di calcolo su un video lungo e denso.
                pluck_density_min = st.slider("Densità Minima (pizzicate/finestra)", 0, 5, 0, key='pluck_dens_min')
                pluck_density_max = st.slider("Densità Massima (pizzicate/finestra)", 0, 5, 2, key='pluck_dens_max')
                pluck_pitch_min = st.slider("Intonazione Minima (Hz)", 40, 800, 80, key='pluck_pitch_min')
                pluck_pitch_max = st.slider("Intonazione Massima (Hz)", 40, 1200, 400, key='pluck_pitch_max')
                pluck_amp_min = st.slider("Ampiezza Minima", 0.0, 1.0, 0.3, step=0.01, key='pluck_amp_min')
                pluck_amp_max = st.slider("Ampiezza Massima", 0.0, 1.0, 1.0, step=0.01, key='pluck_amp_max')
                pluck_duration = st.slider(
                    "Durata Massima Pizzicata (sec)", 0.1, 1.0, 0.4, step=0.05, key='pluck_duration',
                    help="Quanto a lungo può risuonare una corda prima di essere tagliata."
                )

                st.markdown("##### Carattere")
                pluck_hardness = st.slider(
                    "Durezza Eccitazione (pizzicata → martellata)", 0.0, 1.0, 0.0, step=0.05, key='pluck_hardness',
                    help="0 = pizzicata classica (rumore bianco, morbida). Valori più alti = attacco "
                         "più duro e percussivo, più vicino a un martelletto di pianoforte/clavicembalo."
                )
                pluck_unison_voices = st.select_slider(
                    "Corde all'Unisono", options=[1, 2, 3], value=1, key='pluck_unison_voices',
                    help="Più di 1 somma 2-3 corde leggermente scordate per nota (come le corde "
                         "multiple di un pianoforte vero), per un carattere più ricco/mosso invece "
                         "di una pizzicata singola e statica. Aumenta il tempo di generazione."
                )
                if pluck_unison_voices > 1:
                    pluck_unison_detune = st.slider(
                        "Scordatura Unisono (cent)", 1.0, 25.0, 8.0, step=1.0, key='pluck_unison_detune',
                        help="Quanto le corde extra sono scordate rispetto all'intonazione nominale. "
                             "Valori bassi (5-10) danno un 'battimento' naturale, alti (20+) un chorus più marcato."
                    )
                else:
                    pluck_unison_detune = 0.0

                params['pluck_density_source'] = pluck_density_source
                params['pluck_pitch_source'] = pluck_pitch_source
                params['pluck_damping_source'] = pluck_damping_source
                params['pluck_amp_source'] = pluck_amp_source
                params['pluck_density_range'] = (pluck_density_min, pluck_density_max)
                params['pluck_pitch_range'] = (pluck_pitch_min, pluck_pitch_max)
                params['pluck_amp_range'] = (pluck_amp_min, pluck_amp_max)
                params['pluck_duration'] = pluck_duration
                params['pluck_hardness'] = pluck_hardness
                params['pluck_unison_voices'] = pluck_unison_voices
                params['pluck_unison_detune'] = pluck_unison_detune

                def _pluck_source_data(name):
                    if name == "Luminosità": return luminosity_data
                    if name == "Dettaglio": return detail_data
                    if name == "Movimento": return movement_data
                    if name == "Densità Contorni": return edge_density_data
                    if name == "Variazione Colore": return color_variation_data
                    return []

                pluck_density_data_raw = _pluck_source_data(pluck_density_source)
                pluck_pitch_data_raw = _pluck_source_data(pluck_pitch_source)
                pluck_damping_data_raw = _pluck_source_data(pluck_damping_source)
                pluck_amp_data_raw = _pluck_source_data(pluck_amp_source)

                pluck_density_scaled = np.interp(pluck_density_data_raw, (min(pluck_density_data_raw) if pluck_density_data_raw else 0, max(pluck_density_data_raw) if pluck_density_data_raw else 1), (pluck_density_min, pluck_density_max)).tolist()
                pluck_pitch_scaled = scale_frequency_exponential(pluck_pitch_data_raw, pluck_pitch_min, pluck_pitch_max)
                pluck_pitch_scaled = apply_melody(pluck_pitch_scaled)
                # Smorzamento normalizzato in [0,1] indipendentemente dal range della sorgente scelta
                pluck_damping_scaled = np.interp(pluck_damping_data_raw, (min(pluck_damping_data_raw) if pluck_damping_data_raw else 0, max(pluck_damping_data_raw) if pluck_damping_data_raw else 1), (0.0, 1.0)).tolist()
                pluck_amp_scaled = np.interp(pluck_amp_data_raw, (min(pluck_amp_data_raw) if pluck_amp_data_raw else 0, max(pluck_amp_data_raw) if pluck_amp_data_raw else 1), (pluck_amp_min, pluck_amp_max)).tolist()
            else:
                pluck_layer_gain = 0.0
                pluck_density_scaled = []
                pluck_pitch_scaled = []
                pluck_damping_scaled = []
                pluck_amp_scaled = []
                pluck_duration = 0.4
                pluck_hardness = 0.0
                pluck_unison_voices = 1
                pluck_unison_detune = 0.0
                params['pluck_duration'] = pluck_duration
                params['pluck_hardness'] = pluck_hardness
                params['pluck_unison_voices'] = pluck_unison_voices
                params['pluck_unison_detune'] = pluck_unison_detune

        # Layer 4: Rumore (Basato su Variazione Movimento)
        with tab_noise:
            st.markdown("### Layer: Rumore")
            use_noise = st.checkbox("Abilita Rumore", value=True, key='noise_on')
            params['noise_enabled'] = use_noise
            if use_noise:
                noise_layer_gain = st.slider("🎚️ Volume Layer Rumore", 0.0, 2.0, 1.0, step=0.05, key='noise_gain',
                                              help="Bilancia questo layer rispetto agli altri attivi contemporaneamente.")
                params['noise_layer_gain'] = noise_layer_gain

                noise_amp_source = st.selectbox("Sorgente Ampiezza Rumore", ["Variazione Movimento", "Movimento", "Dettaglio", "Densità Contorni", "Variazione Colore"], key='noise_amp_src')
                noise_amp_min = st.slider("Ampiezza Minima Rumore", 0.0, 1.0, 0.0, step=0.01, key='noise_amp_min')
                noise_amp_max = st.slider("Ampiezza Massima Rumore", 0.0, 1.0, 0.1, step=0.01, key='noise_amp_max')

                params['noise_amp_source'] = noise_amp_source
                params['noise_amp_range'] = (noise_amp_min, noise_amp_max)

                noise_amp_data_raw = []
                if noise_amp_source == "Variazione Movimento": noise_amp_data_raw = variation_movement_data
                elif noise_amp_source == "Movimento": noise_amp_data_raw = movement_data
                elif noise_amp_source == "Dettaglio": noise_amp_data_raw = detail_data
                elif noise_amp_source == "Densità Contorni": noise_amp_data_raw = edge_density_data
                elif noise_amp_source == "Variazione Colore": noise_amp_data_raw = color_variation_data

                noise_amp_scaled = np.interp(noise_amp_data_raw, (min(noise_amp_data_raw) if noise_amp_data_raw else 0, max(noise_amp_data_raw) if noise_amp_data_raw else 1), (noise_amp_min, noise_amp_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                noise_amp_scaled = []
                noise_layer_gain = 0.0


        # Effetti Audio (Glitch, Delay, Reverb)
        with tab_fx:
            st.markdown("### Effetti Audio")
            
            # Glitch
            st.subheader("Glitch")
            use_glitch = st.checkbox("Abilita Glitch", value=False, key='glitch_on')
            params['glitch_enabled'] = use_glitch
            if use_glitch:
                glitch_character = st.selectbox(
                    "Carattere del Glitch",
                    ["Bilanciato (default)", "Pulito / Digitale (repeat + reverse, poco rumore)", "Sporco / Analogico (rumore dominante)"],
                    key='glitch_character',
                    help="Prima il tipo di glitch (repeat/noise/reverse) era sempre scelto a caso in modo uniforme, "
                         "quindi 'VHS sporco' e 'Datamosh pulito' suonavano uguali nel carattere. Ora puoi pesare la scelta."
                )
                params['glitch_character'] = glitch_character

                glitch_factor_source = st.selectbox("Sorgente Fattore Glitch (Probabilità)", ["Variazione Movimento", "Movimento", "Dettaglio", "Densità Contorni", "Variazione Colore"], key='glitch_factor_src')
                glitch_intensity_source = st.selectbox("Sorgente Intensità Glitch (Durata/Ampiezza)", ["Variazione Movimento", "Movimento", "Dettaglio", "Densità Contorni", "Variazione Colore"], key='glitch_intensity_src')
                
                glitch_factor_min = st.slider("Fattore Minimo Glitch (0-1)", 0.0, 1.0, 0.01, step=0.005, key='glitch_factor_min')
                glitch_factor_max = st.slider("Fattore Massimo Glitch (0-1)", 0.0, 1.0, 0.1, step=0.005, key='glitch_factor_max')
                glitch_intensity_min = st.slider("Intensità Minima Glitch (0-1)", 0.0, 1.0, 0.1, step=0.01, key='glitch_intensity_min')
                glitch_intensity_max = st.slider("Intensità Massima Glitch (0-1)", 0.0, 1.0, 0.8, step=0.01, key='glitch_intensity_max')

                if glitch_character.startswith("Pulito"):
                    glitch_type_weights = {"repeat": 0.45, "noise": 0.1, "reverse": 0.45}
                elif glitch_character.startswith("Sporco"):
                    glitch_type_weights = {"repeat": 0.2, "noise": 0.65, "reverse": 0.15}
                else:
                    glitch_type_weights = {"repeat": 1/3, "noise": 1/3, "reverse": 1/3}

                params['glitch_factor_source'] = glitch_factor_source
                params['glitch_intensity_source'] = glitch_intensity_source
                params['glitch_factor_range'] = (glitch_factor_min, glitch_factor_max)
                params['glitch_intensity_range'] = (glitch_intensity_min, glitch_intensity_max)

                glitch_factor_data_raw = []
                if glitch_factor_source == "Variazione Movimento": glitch_factor_data_raw = variation_movement_data
                elif glitch_factor_source == "Movimento": glitch_factor_data_raw = movement_data
                elif glitch_factor_source == "Dettaglio": glitch_factor_data_raw = detail_data
                elif glitch_factor_source == "Densità Contorni": glitch_factor_data_raw = edge_density_data
                elif glitch_factor_source == "Variazione Colore": glitch_factor_data_raw = color_variation_data

                glitch_intensity_data_raw = []
                if glitch_intensity_source == "Variazione Movimento": glitch_intensity_data_raw = variation_movement_data
                elif glitch_intensity_source == "Movimento": glitch_intensity_data_raw = movement_data
                elif glitch_intensity_source == "Dettaglio": glitch_intensity_data_raw = detail_data
                elif glitch_intensity_source == "Densità Contorni": glitch_intensity_data_raw = edge_density_data
                elif glitch_intensity_source == "Variazione Colore": glitch_intensity_data_raw = color_variation_data

                glitch_factor_scaled = np.interp(glitch_factor_data_raw, (min(glitch_factor_data_raw) if glitch_factor_data_raw else 0, max(glitch_factor_data_raw) if glitch_factor_data_raw else 1), (glitch_factor_min, glitch_factor_max)).tolist()
                glitch_intensity_data = np.interp(glitch_intensity_data_raw, (min(glitch_intensity_data_raw) if glitch_intensity_data_raw else 0, max(glitch_intensity_data_raw) if glitch_intensity_data_raw else 1), (glitch_intensity_min, glitch_intensity_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                glitch_factor_scaled = []
                glitch_intensity_data = []
                glitch_type_weights = {"repeat": 1/3, "noise": 1/3, "reverse": 1/3}

            # Delay
            st.subheader("Delay")
            use_delay = st.checkbox("Abilita Delay", value=False, key='delay_on')
            params['delay_enabled'] = use_delay
            if use_delay:
                delay_time_source = st.selectbox("Sorgente Tempo Delay (sec)", ["Movimento", "Variazione Movimento", "Luminosità", "Densità Contorni", "Variazione Colore"], key='delay_time_src')
                delay_feedback_source = st.selectbox("Sorgente Feedback Delay (0-1)", ["Movimento", "Variazione Movimento", "Dettaglio", "Densità Contorni", "Variazione Colore"], key='delay_feedback_src')
                
                delay_time_min = st.slider("Tempo Minimo Delay (sec)", 0.01, 0.5, 0.1, step=0.01, key='delay_time_min')
                delay_time_max = st.slider("Tempo Massimo Delay (sec)", 0.01, 0.5, 0.3, step=0.01, key='delay_time_max')
                delay_feedback_min = st.slider("Feedback Minimo Delay", 0.0, 0.95, 0.3, step=0.01, key='delay_feedback_min')
                delay_feedback_max = st.slider("Feedback Massimo Delay", 0.0, 0.95, 0.7, step=0.01, key='delay_feedback_max')

                params['delay_time_source'] = delay_time_source
                params['delay_feedback_source'] = delay_feedback_source
                params['delay_time_range'] = (delay_time_min, delay_time_max)
                params['delay_feedback_range'] = (delay_feedback_min, delay_feedback_max)

                delay_time_data_raw = []
                if delay_time_source == "Movimento": delay_time_data_raw = movement_data
                elif delay_time_source == "Variazione Movimento": delay_time_data_raw = variation_movement_data
                elif delay_time_source == "Luminosità": delay_time_data_raw = luminosity_data
                elif delay_time_source == "Densità Contorni": delay_time_data_raw = edge_density_data
                elif delay_time_source == "Variazione Colore": delay_time_data_raw = color_variation_data

                delay_feedback_data_raw = []
                if delay_feedback_source == "Movimento": delay_feedback_data_raw = movement_data
                elif delay_feedback_source == "Variazione Movimento": delay_feedback_data_raw = variation_movement_data
                elif delay_feedback_source == "Dettaglio": delay_feedback_data_raw = detail_data
                elif delay_feedback_source == "Densità Contorni": delay_feedback_data_raw = edge_density_data
                elif delay_feedback_source == "Variazione Colore": delay_feedback_data_raw = color_variation_data

                delay_time_scaled = np.interp(delay_time_data_raw, (min(delay_time_data_raw) if delay_time_data_raw else 0, max(delay_time_data_raw) if delay_time_data_raw else 1), (delay_time_min, delay_time_max)).tolist()
                delay_feedback_scaled = np.interp(delay_feedback_data_raw, (min(delay_feedback_data_raw) if delay_feedback_data_raw else 0, max(delay_feedback_data_raw) if delay_feedback_data_raw else 1), (delay_feedback_min, delay_feedback_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                delay_time_scaled = []
                delay_feedback_scaled = []


            # Reverb
            st.subheader("Riverbero")
            use_reverb = st.checkbox("Abilita Riverbero", value=False, key='reverb_on')
            params['reverb_enabled'] = use_reverb
            if use_reverb:
                reverb_decay_source = st.selectbox("Sorgente Tempo Decadimento (sec)", ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"], key='reverb_decay_src')
                reverb_mix_source = st.selectbox("Sorgente Mix (Wet/Dry)", ["Luminosità", "Dettaglio", "Movimento", "Densità Contorni", "Variazione Colore"], key='reverb_mix_src')
                
                reverb_decay_min = st.slider("Decadimento Minimo (sec)", 0.1, 5.0, 1.0, step=0.1, key='reverb_decay_min')
                reverb_decay_max = st.slider("Decadimento Massimo (sec)", 0.1, 5.0, 3.0, step=0.1, key='reverb_decay_max')
                reverb_mix_min = st.slider("Mix Minimo (0-1)", 0.0, 1.0, 0.2, step=0.01, key='reverb_mix_min')
                reverb_mix_max = st.slider("Mix Massimo (0-1)", 0.0, 1.0, 0.6, step=0.01, key='reverb_mix_max')

                params['reverb_decay_source'] = reverb_decay_source
                params['reverb_mix_source'] = reverb_mix_source
                params['reverb_decay_range'] = (reverb_decay_min, reverb_decay_max)
                params['reverb_mix_range'] = (reverb_mix_min, reverb_mix_max)

                reverb_decay_data_raw = []
                if reverb_decay_source == "Luminosità": reverb_decay_data_raw = luminosity_data
                elif reverb_decay_source == "Dettaglio": reverb_decay_data_raw = detail_data
                elif reverb_decay_source == "Movimento": reverb_decay_data_raw = movement_data
                elif reverb_decay_source == "Densità Contorni": reverb_decay_data_raw = edge_density_data
                elif reverb_decay_source == "Variazione Colore": reverb_decay_data_raw = color_variation_data

                reverb_mix_data_raw = []
                if reverb_mix_source == "Luminosità": reverb_mix_data_raw = luminosity_data
                elif reverb_mix_source == "Dettaglio": reverb_mix_data_raw = detail_data
                elif reverb_mix_source == "Movimento": reverb_mix_data_raw = movement_data
                elif reverb_mix_source == "Densità Contorni": reverb_mix_data_raw = edge_density_data
                elif reverb_mix_source == "Variazione Colore": reverb_mix_data_raw = color_variation_data

                reverb_decay_scaled = np.interp(reverb_decay_data_raw, (min(reverb_decay_data_raw) if reverb_decay_data_raw else 0, max(reverb_decay_data_raw) if reverb_decay_data_raw else 1), (reverb_decay_min, reverb_decay_max)).tolist()
                reverb_mix_scaled = np.interp(reverb_mix_data_raw, (min(reverb_mix_data_raw) if reverb_mix_data_raw else 0, max(reverb_mix_data_raw) if reverb_mix_data_raw else 1), (reverb_mix_min, reverb_mix_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                reverb_decay_scaled = []
                reverb_mix_scaled = []


        # Equalizzatore Dinamico
        with tab_eq:
            st.markdown("### Equalizzatore Dinamico")
            use_eq = st.checkbox("Abilita Equalizzatore", value=False, key='eq_on')
            params['eq_enabled'] = use_eq
            if use_eq:
                eq_low_source = st.selectbox("Sorgente Guadagno Bassi (dB)", ["Luminosità", "Movimento", "Variazione Movimento", "Densità Contorni", "Variazione Colore"], key='eq_low_src')
                eq_mid_source = st.selectbox("Sorgente Guadagno Medi (dB)", ["Dettaglio", "Luminosità", "Movimento", "Densità Contorni", "Variazione Colore"], key='eq_mid_src')
                eq_high_source = st.selectbox("Sorgente Guadagno Alti (dB)", ["Movimento", "Dettaglio", "Variazione Movimento", "Densità Contorni", "Variazione Colore"], key='eq_high_src')
                
                eq_gain_min = st.slider("Guadagno Minimo (dB)", -20.0, 20.0, -10.0, step=0.5, key='eq_gain_min')
                eq_gain_max = st.slider("Guadagno Massimo (dB)", -20.0, 20.0, 10.0, step=0.5, key='eq_gain_max')

                params['eq_low_source'] = eq_low_source
                params['eq_mid_source'] = eq_mid_source
                params['eq_high_source'] = eq_high_source
                params['eq_gain_range'] = (eq_gain_min, eq_gain_max)

                eq_low_data_raw = []
                if eq_low_source == "Luminosità": eq_low_data_raw = luminosity_data
                elif eq_low_source == "Movimento": eq_low_data_raw = movement_data
                elif eq_low_source == "Variazione Movimento": eq_low_data_raw = variation_movement_data
                elif eq_low_source == "Densità Contorni": eq_low_data_raw = edge_density_data
                elif eq_low_source == "Variazione Colore": eq_low_data_raw = color_variation_data

                eq_mid_data_raw = []
                if eq_mid_source == "Dettaglio": eq_mid_data_raw = detail_data
                elif eq_mid_source == "Luminosità": eq_mid_data_raw = luminosity_data
                elif eq_mid_source == "Movimento": eq_mid_data_raw = movement_data
                elif eq_mid_source == "Densità Contorni": eq_mid_data_raw = edge_density_data
                elif eq_mid_source == "Variazione Colore": eq_mid_data_raw = color_variation_data

                eq_high_data_raw = []
                if eq_high_source == "Movimento": eq_high_data_raw = movement_data
                elif eq_high_source == "Dettaglio": eq_high_data_raw = detail_data
                elif eq_high_source == "Variazione Movimento": eq_high_data_raw = variation_movement_data
                elif eq_high_source == "Densità Contorni": eq_high_data_raw = edge_density_data
                elif eq_high_source == "Variazione Colore": eq_high_data_raw = color_variation_data

                eq_low_scaled = np.interp(eq_low_data_raw, (min(eq_low_data_raw) if eq_low_data_raw else 0, max(eq_low_data_raw) if eq_low_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
                eq_mid_scaled = np.interp(eq_mid_data_raw, (min(eq_mid_data_raw) if eq_mid_data_raw else 0, max(eq_mid_data_raw) if eq_mid_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
                eq_high_scaled = np.interp(eq_high_data_raw, (min(eq_high_data_raw) if eq_high_data_raw else 0, max(eq_high_data_raw) if eq_high_data_raw else 1), (eq_gain_min, eq_gain_max)).tolist()
            else: # Aggiunto else per gestire i casi in cui i dati non sono scalati
                eq_low_scaled = []
                eq_mid_scaled = []
                eq_high_scaled = []

        # Panning Stereo (finora il centro di massa orizzontale veniva calcolato ma mai usato)
        with tab_pan:
            st.markdown("### Panning Stereo")
            st.caption("Il centro di massa orizzontale del video (dove si concentra la 'massa' visiva nel frame) può spostare il suono nello spazio stereo: se il soggetto si muove a sinistra o a destra, il suono lo segue.")
            use_panning = st.checkbox("Abilita Panning Stereo", value=False, key='panning_on')
            params['panning_enabled'] = use_panning
            if use_panning:
                pan_source = st.selectbox("Sorgente Panning", ["Centro di Massa Orizzontale", "Movimento", "Variazione Movimento", "Densità Contorni", "Variazione Colore"], key='pan_src',
                                           help="Centro di Massa Orizzontale è la scelta più naturale: segue davvero dove si trova il soggetto nel frame.")
                params['pan_source'] = pan_source

                pan_data_raw = []
                if pan_source == "Centro di Massa Orizzontale": pan_data_raw = horizontal_mass_center_data
                elif pan_source == "Movimento": pan_data_raw = movement_data
                elif pan_source == "Variazione Movimento": pan_data_raw = variation_movement_data
                elif pan_source == "Densità Contorni": pan_data_raw = edge_density_data
                elif pan_source == "Variazione Colore": pan_data_raw = color_variation_data

                # Il centro di massa orizzontale è già normalizzato 0..1 (0=sinistra, 1=destra):
                # per questa sorgente non serve rimappare min/max, altrimenti un soggetto che sta
                # sempre a destra verrebbe "stirato" fino all'estrema sinistra.
                if pan_source == "Centro di Massa Orizzontale":
                    pan_scaled = pan_data_raw
                else:
                    pan_scaled = np.interp(pan_data_raw, (min(pan_data_raw) if pan_data_raw else 0, max(pan_data_raw) if pan_data_raw else 1), (0.0, 1.0)).tolist()
            else:
                pan_scaled = []

            st.markdown("---")
            st.markdown("### Altezza (Su/Giù)")
            st.caption("Lo stereo non ha un canale fisico per 'sopra' e 'sotto': qui si usa un trucco psicoacustico — un suono più chiaro/brillante viene percepito come 'in alto', uno più scuro/attutito come 'in basso'. Basato sul centro di massa verticale del video.")
            use_elevation = st.checkbox("Abilita Simulazione Altezza", value=False, key='elevation_on')
            params['elevation_enabled'] = use_elevation
            if use_elevation:
                elevation_source = st.selectbox("Sorgente Altezza", ["Centro di Massa Verticale", "Luminosità", "Densità Contorni", "Variazione Colore"], key='elevation_src',
                                                 help="Centro di Massa Verticale segue davvero se il soggetto è in alto o in basso nel frame. Le altre sono mappature più creative.")
                params['elevation_source'] = elevation_source

                if elevation_source == "Centro di Massa Verticale":
                    # Già normalizzato 0..1 (0=alto del frame, 1=basso): nessun rimappaggio min/max,
                    # altrimenti un soggetto sempre in alto verrebbe "stirato" fino in basso.
                    elevation_scaled = vertical_mass_center_data
                elif elevation_source == "Luminosità": # video più luminoso = più "in alto" (elevation_data basso = luminosità alta)
                    elevation_scaled = [1.0 - l for l in luminosity_data]
                else: # Densità Contorni / Variazione Colore: mappatura creativa, normalizzata 0..1 senza inversione
                    elevation_raw = edge_density_data if elevation_source == "Densità Contorni" else color_variation_data
                    elevation_scaled = np.interp(elevation_raw, (min(elevation_raw) if elevation_raw else 0, max(elevation_raw) if elevation_raw else 1), (0.0, 1.0)).tolist()
            else:
                elevation_scaled = []


        if not generative_mode:
            st.subheader("Impostazioni Output Video")
            output_resolution_choice = st.selectbox("Formato Video Output", list(FORMAT_RESOLUTIONS.keys()))
            params['output_resolution_choice'] = output_resolution_choice
        else:
            output_resolution_choice = None
            params['output_resolution_choice'] = "N/A (modalità generativa)"

        st.subheader("Impostazioni Output Audio")

        col_audio, col_video = st.columns(2)
        with col_audio:
            normalize_audio = st.checkbox("Normalizza Audio Finale", value=True)
            params['normalize_audio'] = normalize_audio
            use_saturation = st.checkbox(
                "Saturazione Soft (Analogica)", value=True,
                help="Applica una leggera compressione morbida (tanh) prima della normalizzazione: "
                     "addolcisce i picchi improvvisi (tipici quando grani/glitch/riverbero si sommano) "
                     "con un carattere più 'analogico' invece di un clip digitale netto."
            )
            params['use_saturation'] = use_saturation
            if use_saturation:
                saturation_drive = st.slider(
                    "Intensità Saturazione", 1.0, 3.0, 1.5, step=0.1,
                    help="Valori più alti = compressione più marcata dei picchi (più 'calda'/satura)."
                )
                params['saturation_drive'] = saturation_drive
            else:
                params['saturation_drive'] = 1.0
        with col_video:
            if not generative_mode:
                use_original_audio = st.checkbox("Mantieni Audio Originale del Video (Mix con quello generato)", value=False)
                params['use_original_audio'] = use_original_audio
                if use_original_audio:
                    original_audio_mix_level = st.slider("Livello Mix Audio Originale", 0.0, 1.0, 0.5, step=0.01)
                    params['original_audio_mix_level'] = original_audio_mix_level
                else:
                    params['original_audio_mix_level'] = 0.0
            else:
                use_original_audio = False
                params['use_original_audio'] = False
                params['original_audio_mix_level'] = 0.0
                st.caption("🎲 Modalità generativa: nessun audio originale da mantenere (nessun video caricato).")
            export_stems = st.checkbox(
                "Esporta anche gli stems separati (ZIP)", value=False,
                help="Oltre al video, genera uno ZIP con l'audio di ogni layer attivo separato "
                     "(dry, prima degli effetti), da remixare in un DAW."
            )
            params['export_stems'] = export_stems

        def build_combined_audio(return_stems: bool = False):
            """Genera e mixa tutti i layer/effetti audio secondo i parametri già impostati
            nell'interfaccia (chiusura su tutte le variabili '..._scaled' calcolate sopra).
            Condivisa tra l'anteprima audio rapida e il render video finale, per non duplicare
            la stessa logica in due punti che potrebbero disallinearsi nel tempo.

            Se return_stems=True, restituisce anche un dizionario con l'audio 'dry' di ogni
            singolo layer (già pesato dal proprio Volume Layer, ma prima di effetti/mix/master),
            utile per un export stems separato da remixare in un DAW."""
            mixed_audio = np.zeros(audio_generator.total_samples, dtype=np.float32)
            stems = {}

            if use_subtractive:
                subtractive_audio = audio_generator.generate_subtractive_waveform(sub_freq_scaled, sub_amp_scaled, sub_waveform_type, band_limited=sub_band_limited)
                if sub_filter_enabled:
                    subtractive_audio = audio_generator.apply_resonant_filter(subtractive_audio, sub_cutoff_scaled, sub_resonance_scaled)
                subtractive_audio = subtractive_audio * sub_layer_gain
                mixed_audio += subtractive_audio
                if return_stems:
                    stems['sottrattiva'] = subtractive_audio

            if use_fm:
                fm_audio = audio_generator.generate_fm_layer(fm_carrier_scaled, fm_mod_scaled, fm_mod_idx_scaled, fm_amp_scaled) * fm_layer_gain
                mixed_audio += fm_audio
                if return_stems:
                    stems['fm'] = fm_audio

            if use_epiano:
                epiano_audio = audio_generator.generate_fm_epiano_layer(
                    epiano_density_scaled, epiano_pitch_scaled, epiano_brightness_scaled, epiano_amp_scaled,
                    note_duration=epiano_note_duration, mod_ratio=epiano_mod_ratio
                ) * epiano_layer_gain
                mixed_audio += epiano_audio
                if return_stems:
                    stems['epiano'] = epiano_audio

            if use_granular:
                granular_audio = audio_generator.generate_granular_layer(gran_density_scaled, gran_duration_scaled, gran_amp_scaled, gran_pitch_scaled) * gran_layer_gain
                mixed_audio += granular_audio
                if return_stems:
                    stems['granulare'] = granular_audio

            if use_pluck:
                pluck_audio = audio_generator.generate_pluck_layer(
                    pluck_density_scaled, pluck_pitch_scaled, pluck_damping_scaled, pluck_amp_scaled,
                    max_pluck_duration=pluck_duration, hammer_hardness=pluck_hardness,
                    unison_voices=pluck_unison_voices, unison_detune_cents=pluck_unison_detune
                ) * pluck_layer_gain
                mixed_audio += pluck_audio
                if return_stems:
                    stems['corde'] = pluck_audio

            if use_noise:
                noise_audio = audio_generator.add_noise_layer(noise_amp_scaled) * noise_layer_gain
                mixed_audio += noise_audio
                if return_stems:
                    stems['rumore'] = noise_audio

            if use_glitch:
                mixed_audio = audio_generator.apply_glitch_effect(mixed_audio, glitch_factor_scaled, glitch_intensity_data, glitch_type_weights)

            if use_delay:
                mixed_audio = audio_generator.apply_delay_effect(mixed_audio, delay_time_scaled, delay_feedback_scaled)

            if use_reverb:
                mixed_audio = audio_generator.apply_reverb_effect(mixed_audio, reverb_decay_scaled, reverb_mix_scaled)

            if use_eq:
                mixed_audio = audio_generator.apply_eq_effect(mixed_audio, eq_low_scaled, eq_mid_scaled, eq_high_scaled)

            # Altezza (su/giù): applicata prima del panning, mentre l'audio è ancora mono
            # (illusione timbrica, non di posizione).
            if use_elevation:
                mixed_audio = audio_generator.apply_elevation_filter(mixed_audio, elevation_scaled)

            # Panning stereo: applicato per ultimo, dopo tutti gli effetti mono, così converte
            # l'audio in stereo (samples, 2) solo alla fine della catena.
            if use_panning:
                mixed_audio = audio_generator.apply_stereo_panning(mixed_audio, pan_scaled)

            if use_saturation:
                # Saturazione soft (tanh) prima della normalizzazione: comprime morbidamente i
                # picchi (es. quando più grani/glitch/riverbero si sovrappongono nello stesso
                # istante) invece di lasciare che un clip rigido più avanti li tagli di netto.
                # tanh(x) è per costruzione sempre in (-1, 1) qualunque sia l'ampiezza in ingresso:
                # 'drive' controlla quanto la non-linearità entra in gioco (più alto = più
                # compressione/calore) e, come effetto collaterale voluto, anche il volume
                # percepito dei passaggi più quieti (comportamento standard per un saturatore
                # in stile analogico/nastro).
                mixed_audio = np.tanh(mixed_audio * saturation_drive)

            if normalize_audio:
                # Prevenire divisione per zero se l'audio è silenzioso.
                # Usiamo il picco GLOBALE (non per-canale) per non alterare il bilanciamento
                # stereo introdotto dal panning: normalizzare i due canali in modo indipendente
                # appiattirebbe la differenza di volume tra sinistra e destra.
                peak = np.max(np.abs(mixed_audio))
                if peak > 1e-6:
                    mixed_audio = mixed_audio / peak
                else:
                    mixed_audio = np.zeros_like(mixed_audio) # Mantieni a zero se già silenzioso

            # Assicurati che l'audio sia nel range [-1, 1] per soundfile
            mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

            if return_stems:
                stems = {name: np.clip(stem, -1.0, 1.0) for name, stem in stems.items()}
                return mixed_audio, stems
            return mixed_audio

        if st.button("🔊 Anteprima Audio (senza video)"):
            # Genera solo l'audio, senza toccare FFmpeg/video: pensata per iterare rapidamente
            # sui parametri ad orecchio, senza aspettare ogni volta il render+mux del video intero.
            st.info("🎵 Generazione anteprima audio in corso...")
            preview_audio = build_combined_audio()
            preview_audio_path = f"preview_audio_{session_id}.wav"
            sf.write(preview_audio_path, preview_audio, AUDIO_SAMPLE_RATE)
            st.audio(preview_audio_path, format="audio/wav")
            gc.collect()

        if st.button("🎵 Genera Traccia Audio" if generative_mode else "Genera Video con Audio"):
            # Sempre tenta la generazione audio se il pulsante è cliccato
            st.info("🎵 Generazione e mixaggio audio in corso... Attendere.")
            progress_bar_audio = st.progress(0)
            status_text_audio = st.empty()

            progress_bar_audio.progress(10)
            status_text_audio.text("Generazione layer e applicazione effetti...")

            combined_audio, generated_stems = build_combined_audio(return_stems=True)

            progress_bar_audio.progress(90)
            status_text_audio.text("Scrittura file audio...")

            # audio_output_path è assegnato qui, sempre se il pulsante è cliccato
            audio_output_path = f"output_audio_{session_id}.wav"
            sf.write(audio_output_path, combined_audio, AUDIO_SAMPLE_RATE)

            stems_zip_path = None
            if export_stems and generated_stems:
                stems_zip_path = f"stems_{session_id}.zip"
                with zipfile.ZipFile(stems_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for stem_name, stem_audio in generated_stems.items():
                        stem_wav_path = f"stem_{session_id}_{stem_name}.wav"
                        sf.write(stem_wav_path, stem_audio, AUDIO_SAMPLE_RATE)
                        zf.write(stem_wav_path, arcname=f"{stem_name}.wav")
                        os.remove(stem_wav_path)  # il wav intermedio serve solo per scrivere nello zip
                # Salva i bytes in session_state (stesso pattern del video finale), così il
                # download persiste tra i rerun anche se il file su disco viene poi ripulito.
                with open(stems_zip_path, "rb") as f:
                    st.session_state['stems_zip_bytes'] = f.read()
                os.remove(stems_zip_path)
            else:
                st.session_state['stems_zip_bytes'] = None
            
            progress_bar_audio.progress(100)
            status_text_audio.text("Audio generato!")
            st.success("✅ Audio generato con successo!")
            
            gc.collect() # Libera memoria

            # In modalità generativa non c'è nessun video da unire: offri direttamente l'audio,
            # riusando lo stesso percorso già previsto per quando FFmpeg non è disponibile
            # (stessa logica, messaggio diverso per non confondere "niente video" con un errore).
            if generative_mode:
                st.success(f"🎵 Traccia generativa pronta: '{base_name_output}'.")
                with open(audio_output_path, "rb") as f:
                    audio_bytes = f.read()
                st.session_state['generative_audio_bytes'] = audio_bytes
                st.download_button(
                    "⬇️ Scarica Traccia Audio (WAV)",
                    audio_bytes,
                    file_name=f"{base_name_output}.wav",
                    mime="audio/wav",
                    key="dl_generative_audio_inline"
                )
                if os.path.exists(audio_output_path):
                    os.remove(audio_output_path)
            elif not check_ffmpeg():
                st.warning(f"⚠️ FFmpeg non è installato o non è nel PATH. Impossibile unire il video con l'audio. L'audio generato è disponibile in '{audio_output_path}'.")
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV temporaneo)",
                        f,
                        file_name=f"videosound_generato_audio_{base_name_output}.wav",
                        mime="audio/wav"
                    )
                # Pulisci i file temporanei ANCHE se FFmpeg non è stato trovato
                for temp_f in [video_input_path, audio_output_path]:
                    if temp_f and os.path.exists(temp_f):
                        os.remove(temp_f)
                st.info("🗑️ File temporanei puliti.")
            else: # FFmpeg è disponibile, procedi con l'unione di video e audio
                st.info("🎥 Unione audio/video e ricodifica in corso... Potrebbe richiedere del tempo.")
                progress_bar_video = st.progress(0)
                status_text_video = st.empty()

                final_video_path = f"output_{base_name_output}_{session_id}_{output_resolution_choice.replace(' ', '_')}.mp4"
                
                ffmpeg_command = ["ffmpeg", "-y"]

                temp_original_audio_path = None
                if use_original_audio:
                    # Estrai audio originale. Il video caricato potrebbe non avere una
                    # traccia audio (es. screen recording muto) o avere un codec che
                    # ffmpeg non riesce a rimappare in AAC: in quel caso non crashare,
                    # avvisa l'utente e prosegui usando solo l'audio generato.
                    temp_original_audio_path = f"temp_original_audio_{session_id}.aac"
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", video_input_path, "-vn", "-acodec", "aac", temp_original_audio_path
                        ], check=True, capture_output=True)
                    except subprocess.CalledProcessError:
                        st.warning("⚠️ Impossibile estrarre l'audio originale dal video (potrebbe non avere una traccia audio). Procedo usando solo l'audio generato.")
                        use_original_audio = False
                        params['use_original_audio'] = False
                        if os.path.exists(temp_original_audio_path):
                            os.remove(temp_original_audio_path)
                        temp_original_audio_path = None

                if use_original_audio:
                    # Mix e ricodifica
                    ffmpeg_command.extend([
                        "-i", video_input_path,
                        "-i", audio_output_path,
                        "-i", temp_original_audio_path,
                        "-filter_complex",
                        f"[1:a]volume=1.0[generated_audio];" # volume fisso per audio generato
                        f"[2:a]volume={original_audio_mix_level}[original_audio];" # volume per audio originale
                        f"[generated_audio][original_audio]amix=inputs=2:duration=longest[aout]", # mix
                        "-map", "0:v",
                        "-map", "[aout]",
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
                    
                    # Cerca il progresso corrente nell'output di FFmpeg (la durata totale
                    # è già nota da duration_seconds, quindi non serve riestrarla dallo stderr)
                    time_pattern = r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})"
                    
                    total_seconds = duration_seconds # Già calcolato prima

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
                                    progress_bar_video.progress(min(progress, 99)) # Non arrivare al 100% finché non è finito
                                    status_text_video.text(f"Elaborazione video: {current_seconds:.2f}/{total_seconds:.2f}s")
                    
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)

                    progress_bar_video.progress(100)
                    status_text_video.text("Video completato!")
                    st.success("✅ Video con audio generato con successo! Scarica qui sotto:")
                    
                    # Salva video in session_state (così il download non riavvia la generazione)
                    with open(final_video_path, "rb") as f:
                        st.session_state['video_bytes'] = f.read()
                    preset_slug = preset_to_filename_slug(st.session_state.get('preset_choice', PRESET_MANUALE))
                    name_suffix = f"_{preset_slug}" if preset_slug else ""
                    st.session_state['video_filename'] = f"{base_name_output}{name_suffix}_video.mp4"
                    st.session_state['report_filename'] = f"{base_name_output}{name_suffix}_report.txt"

                    # Costruisci il report in stile archivio Loop507: campi bilingue IT/EN
                    # con prefisso "::", coerente con la documentazione degli altri progetti.
                    def field(it_label: str, en_label: str, value, indent: str = "") -> str:
                        return f"{indent}:: {it_label} / {en_label} :: {value}"

                    def onoff(flag: bool) -> str:
                        return "abilitata / enabled" if flag else "disabilitata / disabled"

                    report_lines = []
                    report_lines.append(":: VIDEOSOUND GEN — REPORT")
                    report_lines.append(":: by Loop507")
                    report_lines.append("")

                    preset_used = st.session_state.get('preset_choice', PRESET_MANUALE)
                    report_lines.append(field("preset di partenza", "starting preset", preset_used))
                    report_lines.append(field("formato output video", "video output format", params.get('output_resolution_choice', 'N/A')))
                    if params.get('use_original_audio'):
                        report_lines.append(field("audio originale", "original audio", f"mantenuto / kept (mix {params.get('original_audio_mix_level', 0):.2f})"))
                    else:
                        report_lines.append(field("audio originale", "original audio", "non mantenuto / not kept"))
                    report_lines.append(field("normalizzazione audio finale", "final audio normalization", "sì / yes" if params.get('normalize_audio') else "no / no"))
                    if params.get('use_saturation'):
                        report_lines.append(field("saturazione soft", "soft saturation", f"abilitata / enabled (drive {params.get('saturation_drive', 1.5):.1f})"))
                    else:
                        report_lines.append(field("saturazione soft", "soft saturation", "disabilitata / disabled"))
                    if params.get('melody_enabled'):
                        report_lines.append(field("melodia (quantizzazione a scala)", "melody (scale quantization)", f"{params.get('melody_scale', 'N/A')} (fondamentale {params.get('melody_root_note', 220.0):.0f} Hz)"))
                    else:
                        report_lines.append(field("melodia (quantizzazione a scala)", "melody (scale quantization)", "disabilitata / disabled (intonazione libera / free pitch)"))

                    report_lines.append("")
                    report_lines.append(":: --- layer audio / audio layers ---")

                    report_lines.append(field("sintesi sottrattiva", "subtractive synthesis", onoff(params.get('subtractive_enabled'))))
                    if params.get('subtractive_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('sub_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("frequenza", "frequency", f"{params['sub_freq_source']} ({params['sub_freq_range'][0]}-{params['sub_freq_range'][1]} Hz)", indent="  "))
                        report_lines.append(field("ampiezza", "amplitude", f"{params['sub_amp_source']} ({params['sub_amp_range'][0]:.2f}-{params['sub_amp_range'][1]:.2f})", indent="  "))
                        report_lines.append(field("forma d'onda", "waveform", params['sub_waveform_type'], indent="  "))
                        if params['sub_waveform_type'] in ("square", "sawtooth"):
                            report_lines.append(field("anti-aliasing", "anti-aliasing", "band-limited (PolyBLEP)" if params.get('sub_band_limited') else "grezzo / raw (aliasing voluto)", indent="  "))
                        if params.get('sub_filter_enabled'):
                            report_lines.append(field("filtro risonante", "resonant filter", f"cutoff {params.get('sub_cutoff_source', 'N/A')} ({params['sub_cutoff_range'][0]}-{params['sub_cutoff_range'][1]} Hz), Q {params.get('sub_resonance', 1.5):.1f}", indent="  "))
                        else:
                            report_lines.append(field("filtro risonante", "resonant filter", "disabilitato / disabled", indent="  "))

                    report_lines.append(field("sintesi fm", "fm synthesis", onoff(params.get('fm_enabled'))))
                    if params.get('fm_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('fm_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("portante", "carrier", f"{params['fm_carrier_source']} ({params['fm_carrier_range'][0]}-{params['fm_carrier_range'][1]} Hz)", indent="  "))
                        report_lines.append(field("modulatore", "modulator", f"{params['fm_mod_source']} ({params['fm_mod_range'][0]}-{params['fm_mod_range'][1]} Hz)", indent="  "))
                        report_lines.append(field("indice mod.", "mod. index", f"{params['fm_mod_idx_source']} ({params['fm_mod_idx_range'][0]:.1f}-{params['fm_mod_idx_range'][1]:.1f})", indent="  "))
                        report_lines.append(field("ampiezza", "amplitude", f"{params['fm_amp_source']} ({params['fm_amp_range'][0]:.2f}-{params['fm_amp_range'][1]:.2f})", indent="  "))

                    report_lines.append(field("e-piano (fm a note)", "e-piano (fm notes)", onoff(params.get('epiano_enabled'))))
                    if params.get('epiano_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('epiano_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("densità", "density", f"{params.get('epiano_density_source', 'N/A')} ({params['epiano_density_range'][0]}-{params['epiano_density_range'][1]} note/finestra)", indent="  "))
                        report_lines.append(field("intonazione", "pitch", f"{params.get('epiano_pitch_source', 'N/A')} ({params['epiano_pitch_range'][0]}-{params['epiano_pitch_range'][1]} Hz)", indent="  "))
                        report_lines.append(field("brillantezza", "brightness", params.get('epiano_brightness_source', 'N/A'), indent="  "))
                        report_lines.append(field("rapporto mod/portante", "mod/carrier ratio", f"{params.get('epiano_mod_ratio', 1.0):.1f}", indent="  "))
                        report_lines.append(field("durata massima nota", "max note duration", f"{params.get('epiano_note_duration', 1.2):.2f} sec", indent="  "))

                    report_lines.append(field("sintesi granulare", "granular synthesis", onoff(params.get('granular_enabled'))))
                    if params.get('granular_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('gran_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("densità", "density", f"{params['gran_density_source']} ({params['gran_density_range'][0]}-{params['gran_density_range'][1]} grani/grains)", indent="  "))
                        report_lines.append(field("durata", "duration", f"{params['gran_duration_source']} ({params['gran_duration_range'][0]:.3f}-{params['gran_duration_range'][1]:.3f} sec)", indent="  "))
                        report_lines.append(field("ampiezza", "amplitude", f"{params['gran_amp_source']} ({params['gran_amp_range'][0]:.2f}-{params['gran_amp_range'][1]:.2f})", indent="  "))
                        if 'gran_pitch_range' in params:
                            report_lines.append(field("intonazione grani", "grain pitch", f"{params.get('gran_pitch_source', 'N/A')} ({params['gran_pitch_range'][0]}-{params['gran_pitch_range'][1]} Hz)", indent="  "))

                    report_lines.append(field("corde (karplus-strong)", "strings (karplus-strong)", onoff(params.get('pluck_enabled'))))
                    if params.get('pluck_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('pluck_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("densità", "density", f"{params.get('pluck_density_source', 'N/A')} ({params['pluck_density_range'][0]}-{params['pluck_density_range'][1]} pizzicate/finestra)", indent="  "))
                        report_lines.append(field("intonazione", "pitch", f"{params.get('pluck_pitch_source', 'N/A')} ({params['pluck_pitch_range'][0]}-{params['pluck_pitch_range'][1]} Hz)", indent="  "))
                        report_lines.append(field("smorzamento", "damping", params.get('pluck_damping_source', 'N/A'), indent="  "))
                        report_lines.append(field("durata massima", "max duration", f"{params.get('pluck_duration', 0.4):.2f} sec", indent="  "))
                        report_lines.append(field("carattere", "character", f"durezza {params.get('pluck_hardness', 0.0):.2f} (0=pizzicata, 1=martellata)", indent="  "))
                        if params.get('pluck_unison_voices', 1) > 1:
                            report_lines.append(field("unisono", "unison", f"{params['pluck_unison_voices']} corde, scordatura {params.get('pluck_unison_detune', 0.0):.0f} cent", indent="  "))
                        else:
                            report_lines.append(field("unisono", "unison", "disabilitato / disabled (1 corda)", indent="  "))

                    report_lines.append(field("rumore", "noise", onoff(params.get('noise_enabled'))))
                    if params.get('noise_enabled'):
                        report_lines.append(field("volume layer", "layer volume", f"{params.get('noise_layer_gain', 1.0):.2f}", indent="  "))
                        report_lines.append(field("ampiezza", "amplitude", f"{params['noise_amp_source']} ({params['noise_amp_range'][0]:.2f}-{params['noise_amp_range'][1]:.2f})", indent="  "))

                    report_lines.append("")
                    report_lines.append(":: --- effetti audio / audio effects ---")

                    report_lines.append(field("glitch", "glitch", onoff(params.get('glitch_enabled'))))
                    if params.get('glitch_enabled'):
                        report_lines.append(field("carattere", "character", params.get('glitch_character', 'Bilanciato (default)'), indent="  "))
                        report_lines.append(field("fattore", "factor", f"{params['glitch_factor_source']} ({params['glitch_factor_range'][0]:.3f}-{params['glitch_factor_range'][1]:.3f})", indent="  "))
                        report_lines.append(field("intensità", "intensity", f"{params['glitch_intensity_source']} ({params['glitch_intensity_range'][0]:.2f}-{params['glitch_intensity_range'][1]:.2f})", indent="  "))

                    report_lines.append(field("delay", "delay", onoff(params.get('delay_enabled'))))
                    if params.get('delay_enabled'):
                        report_lines.append(field("tempo", "time", f"{params['delay_time_source']} ({params['delay_time_range'][0]:.2f}-{params['delay_time_range'][1]:.2f} sec)", indent="  "))
                        report_lines.append(field("feedback", "feedback", f"{params['delay_feedback_source']} ({params['delay_feedback_range'][0]:.2f}-{params['delay_feedback_range'][1]:.2f})", indent="  "))

                    report_lines.append(field("riverbero", "reverb", onoff(params.get('reverb_enabled'))))
                    if params.get('reverb_enabled'):
                        report_lines.append(field("decadimento", "decay", f"{params['reverb_decay_source']} ({params['reverb_decay_range'][0]:.1f}-{params['reverb_decay_range'][1]:.1f} sec)", indent="  "))
                        report_lines.append(field("mix", "mix", f"{params['reverb_mix_source']} ({params['reverb_mix_range'][0]:.2f}-{params['reverb_mix_range'][1]:.2f})", indent="  "))

                    report_lines.append(field("equalizzatore dinamico", "dynamic equalizer", onoff(params.get('eq_enabled'))))
                    if params.get('eq_enabled'):
                        report_lines.append(field("bassi", "low", f"{params['eq_low_source']} ({params['eq_gain_range'][0]:.1f}-{params['eq_gain_range'][1]:.1f} dB)", indent="  "))
                        report_lines.append(field("medi", "mid", f"{params['eq_mid_source']} ({params['eq_gain_range'][0]:.1f}-{params['eq_gain_range'][1]:.1f} dB)", indent="  "))
                        report_lines.append(field("alti", "high", f"{params['eq_high_source']} ({params['eq_gain_range'][0]:.1f}-{params['eq_gain_range'][1]:.1f} dB)", indent="  "))

                    report_lines.append("")
                    report_lines.append(":: --- spazializzazione / spatialization ---")
                    if params.get('panning_enabled'):
                        report_lines.append(field("panning stereo", "stereo panning", f"abilitato / enabled (sorgente/source: {params.get('pan_source', 'N/A')})"))
                    else:
                        report_lines.append(field("panning stereo", "stereo panning", "disabilitato / disabled"))
                    if params.get('elevation_enabled'):
                        report_lines.append(field("simulazione altezza", "elevation simulation", f"abilitata / enabled (sorgente/source: {params.get('elevation_source', 'N/A')})"))
                    else:
                        report_lines.append(field("simulazione altezza", "elevation simulation", "disabilitata / disabled"))

                    report_lines.append("")
                    report_lines.append("#loop507 #videosoundgen #glitchbrutalista #minimalismocomputazionale")

                    st.session_state['report_text'] = "\n".join(report_lines)
                    
                    # Pulisci i file temporanei
                    for temp_f in [video_input_path, audio_output_path, temp_original_audio_path if use_original_audio else None, final_video_path]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")

                except subprocess.CalledProcessError as e:
                    st.error(f"❌ Errore FFmpeg durante l'unione/ricodifica: {e.stderr.decode()}")
                    st.code(e.stdout.decode() + e.stderr.decode())
                    # Pulisci i file temporanei anche in caso di errore FFmpeg
                    for temp_f in [video_input_path, audio_output_path, temp_original_audio_path if use_original_audio else None]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")
                except Exception as e:
                    st.error(f"❌ Errore generico durante l'unione/ricodifica: {str(e)}")
                    # Pulisci i file temporanei anche in caso di errore generico
                    for temp_f in [video_input_path, audio_output_path, temp_original_audio_path if use_original_audio else None]:
                        if temp_f and os.path.exists(temp_f):
                            os.remove(temp_f)
                    st.info("🗑️ File temporanei puliti.")
        # Non c'è più un blocco 'else' diretto per 'if st.button' che usi audio_output_path
        # La logica del "FFmpeg non trovato" è ora gestita all'interno del blocco 'if st.button'
        # Questo previene l'UnboundLocalError quando il pulsante non è ancora stato cliccato.


        # ── PULSANTI DI DOWNLOAD PERSISTENTI ───────────────────────────────────
        # Sono fuori dal blocco if st.button → vengono renderizzati ad ogni rerun
        # senza riavviare la generazione. I dati vengono letti da session_state.
        if st.session_state.get('video_bytes') or st.session_state.get('report_text') or st.session_state.get('stems_zip_bytes') or st.session_state.get('generative_audio_bytes'):
            st.markdown("---")
            st.subheader("⬇️ Download")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                if st.session_state.get('video_bytes'):
                    st.download_button(
                        label="🎬 Scarica Video Finale",
                        data=st.session_state['video_bytes'],
                        file_name=st.session_state.get('video_filename', 'output_video.mp4'),
                        mime="video/mp4",
                        key="dl_video"
                    )
                if st.session_state.get('generative_audio_bytes'):
                    st.download_button(
                        label="🎵 Scarica Traccia Audio",
                        data=st.session_state['generative_audio_bytes'],
                        file_name=f"{base_name_output}.wav" if base_name_output else "traccia_generativa.wav",
                        mime="audio/wav",
                        key="dl_generative_audio"
                    )
            with dl_col2:
                if st.session_state.get('report_text'):
                    st.download_button(
                        label="📄 Scarica Report Parametri",
                        data=st.session_state['report_text'],
                        file_name=st.session_state.get('report_filename', f"{base_name_output}_report.txt"),
                        mime="text/plain",
                        key="dl_report"
                    )
            with dl_col3:
                if st.session_state.get('stems_zip_bytes'):
                    st.download_button(
                        label="🎛️ Scarica Stems (ZIP)",
                        data=st.session_state['stems_zip_bytes'],
                        file_name=f"{base_name_output}_stems.zip",
                        mime="application/zip",
                        key="dl_stems"
                    )


        # Questa sezione è volutamente fuori dal blocco if/else del pulsante,
        # ma all'interno dell'if uploaded_file is not None.
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
            if params['subtractive_enabled']:
                st.markdown("##### Sintesi Sottrattiva (Abilitata):")
                st.write(f"- Frequenza Controllata da: **{params['sub_freq_source']}** ({params['sub_freq_range'][0]} - {params['sub_freq_range'][1]} Hz)")
                st.write(f"- Ampiezza Controllata da: **{params['sub_amp_source']}** ({params['sub_amp_range'][0]:.2f} - {params['sub_amp_range'][1]:.2f})")
                st.write(f"- Tipo Onda: **{params['sub_waveform_type']}**")
            else:
                st.write("##### Sintesi Sottrattiva: Disabilitata")

            if params['fm_enabled']:
                st.markdown("##### Sintesi FM (Abilitata):")
                st.write(f"- Frequenza Portante Controllata da: **{params['fm_carrier_source']}** ({params['fm_carrier_range'][0]} - {params['fm_carrier_range'][1]} Hz)")
                st.write(f"- Frequenza Modulatore Controllata da: **{params['fm_mod_source']}** ({params['fm_mod_range'][0]} - {params['fm_mod_range'][1]} Hz)")
                st.write(f"- Indice Modulazione Controllato da: **{params['fm_mod_idx_source']}** ({params['fm_mod_idx_range'][0]:.1f} - {params['fm_mod_idx_range'][1]:.1f})")
                st.write(f"- Ampiezza FM Controllata da: **{params['fm_amp_source']}** ({params['fm_amp_range'][0]:.2f} - {params['fm_amp_range'][1]:.2f})")
            else:
                st.write("##### Sintesi FM: Disabilitata")

            if params['granular_enabled']:
                st.markdown("##### Sintesi Granulare (Abilitata):")
                st.write(f"- Densità Grani Controllata da: **{params['gran_density_source']}** ({params['gran_density_range'][0]} - {params['gran_density_range'][1]} grani)")
                st.write(f"- Durata Grani Controllata da: **{params['gran_duration_source']}** ({params['gran_duration_range'][0]:.3f} - {params['gran_duration_range'][1]:.3f} sec)")
                st.write(f"- Ampiezza Grani Controllata da: **{params['gran_amp_source']}** ({params['gran_amp_range'][0]:.2f} - {params['gran_amp_range'][1]:.2f})")
            else:
                st.write("##### Sintesi Granulare: Disabilitata")

            if params['noise_enabled']:
                st.markdown("##### Rumore (Abilitato):")
                st.write(f"- Ampiezza Rumore Controllata da: **{params['noise_amp_source']}** ({params['noise_amp_range'][0]:.2f} - {params['noise_amp_range'][1]:.2f})")
            else:
                st.write("##### Rumore: Disabilitato")

            st.markdown("#### Effetti Audio:")
            if params['glitch_enabled']:
                st.markdown("##### Glitch (Abilitato):")
                st.write(f"- Fattore Glitch (Probabilità) Controllato da: **{params['glitch_factor_source']}** ({params['glitch_factor_range'][0]:.3f} - {params['glitch_factor_range'][1]:.3f})")
                st.write(f"- Intensità Glitch (Durata/Ampiezza) Controllata da: **{params['glitch_intensity_source']}** ({params['glitch_intensity_range'][0]:.2f} - {params['glitch_intensity_range'][1]:.2f})")
            else:
                st.write("##### Glitch: Disabilitato")

            if params['delay_enabled']:
                st.markdown("##### Delay (Abilitato):")
                st.write(f"- Tempo Delay Controllato da: **{params['delay_time_source']}** ({params['delay_time_range'][0]:.2f} - {params['delay_time_range'][1]:.2f} sec)")
                st.write(f"- Feedback Delay Controllato da: **{params['delay_feedback_source']}** ({params['delay_feedback_range'][0]:.2f} - {params['delay_feedback_range'][1]:.2f})")
            else:
                st.write("##### Delay: Disabilitato")

            if params['reverb_enabled']:
                st.markdown("##### Riverbero (Abilitato):")
                st.write(f"- Tempo Decadimento Controllato da: **{params['reverb_decay_source']}** ({params['reverb_decay_range'][0]:.1f} - {params['reverb_decay_range'][1]:.1f} sec)")
                st.write(f"- Mix Riverbero Controllato da: **{params['reverb_mix_source']}** ({params['reverb_mix_range'][0]:.2f} - {params['reverb_mix_range'][1]:.2f})")
            else:
                st.write("##### Riverbero: Disabilitato")

            if params['eq_enabled']:
                st.markdown("##### Equalizzatore Dinamico (Abilitato):")
                st.write(f"- Guadagno Bassi Controllato da: **{params['eq_low_source']}** ({params['eq_gain_range'][0]:.1f} - {params['eq_gain_range'][1]:.1f} dB)")
                st.write(f"- Guadagno Medi Controllato da: **{params['eq_mid_source']}** ({params['eq_gain_range'][0]:.1f} - {params['eq_gain_range'][1]:.1f} dB)")
                st.write(f"- Guadagno Alti Controllato da: **{params['eq_high_source']}** ({params['eq_gain_range'][0]:.1f} - {params['eq_gain_range'][1]:.1f} dB)")
            else:
                st.write("##### Equalizzatore Dinamico: Disabilitato")

    gc.collect() # Questa riga è ora all'interno del blocco `if uploaded_file is not None`

if __name__ == "__main__":
    main()
