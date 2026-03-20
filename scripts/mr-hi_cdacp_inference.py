import ctranslate2
import subword_nmt.apply_bpe as apply_bpe
import os

# TODO: make this configurable later
MODEL_DIR = r"Inlcude path to the model directory"
INPUT_FILE = r"Path to a source .txt file"
OUTPUT_FILE = "Path to an output .txt file"
SOURCE_BPE = r"Path to source BPE codes"
DEVICE = "cpu"  # change to cuda if you have gpu

def load_models():
    # load the translator
    translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE)
    
    # load bpe encoder
    with open(SOURCE_BPE, 'r', encoding='utf-8') as codes_file:
        bpe = apply_bpe.BPE(codes_file)
    
    return translator, bpe

def read_sentences(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                sentences.append(line)
    return sentences

def bpe_encode_batch(sentences, bpe_encoder):
    encoded = []
    for sent in sentences:
        tokens = bpe_encoder.process_line(sent).split()
        encoded.append(tokens)
    return encoded

def bpe_decode_batch(token_lists):
    decoded = []
    for tokens in token_lists:
        # just join with spaces and remove @@
        text = ' '.join(tokens)
        text = text.replace('@@', '')
        decoded.append(text)
    return decoded

# main code
print("Loading models...")
translator, bpe_encoder = load_models()

print(f"Reading input from {INPUT_FILE}")
input_sentences = read_sentences(INPUT_FILE)
print(f"Got {len(input_sentences)} sentences")

# encode with BPE
print("Encoding with BPE...")
bpe_encoded = bpe_encode_batch(input_sentences, bpe_encoder)

print("Translating...")
results = translator.translate_batch(bpe_encoded, beam_size=4)

print("Decoding translations...")
translated_tokens = [result.hypotheses[0] for result in results]
final_translations = bpe_decode_batch(translated_tokens)

print(f"Writing output to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for translation in final_translations:
        f.write(translation + '\n')

print("Done!")
print(f"Translated {len(final_translations)} sentences")