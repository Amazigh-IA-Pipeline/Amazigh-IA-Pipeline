"""Tiny offline TTS (espeak-ng via espeakng-loader) to make speech test audio."""
import ctypes, sys, wave
import espeakng_loader

lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
lib.espeak_ng_InitializePath(espeakng_loader.get_data_path().encode())
assert lib.espeak_ng_Initialize(None) == 0
assert lib.espeak_ng_InitializeOutput(1, 0, None) == 0  # synchronous
SAMPLES = []
CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
def _cb(wav, n, events):
    if n > 0:
        SAMPLES.extend(wav[:n])
    return 0
_cb_ref = CB(_cb)
lib.espeak_SetSynthCallback(_cb_ref)
RATE = lib.espeak_ng_GetSampleRate()

def synth(text, voice='en', speed=160):
    SAMPLES.clear()
    lib.espeak_SetVoiceByName(voice.encode())
    lib.espeak_SetParameter(1, speed, 0)  # espeakRATE
    b = text.encode('utf-8')
    lib.espeak_Synth(b, len(b) + 1, 0, 1, 0, 1, None, None)
    lib.espeak_Synchronize()
    return list(SAMPLES), RATE

if __name__ == '__main__':
    s, r = synth(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'en')
    with wave.open(sys.argv[1], 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(r)
        w.writeframes(b''.join(int(x).to_bytes(2, 'little', signed=True) for x in s))
    print(len(s) / r, 'seconds at', r)
