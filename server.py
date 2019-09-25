from predict import load_model, predict
from utils.seq import extract_candidates
from flask import Flask, request, jsonify
from constants import MAX_LENGTH, START_CODONS, STOP_CODONS, BEST_MODEL_PATH, TRUNCATED, EXTENSION
from tqdm import tqdm


app = Flask(__name__)
model, data_loader = load_model(BEST_MODEL_PATH)


def find_orf(sequence, truncated=False, return_top=None, return_best=True, include_seq=True):
    """
    This function takes a single sequence and returns a list of
    dictionaries in the following format:
    {
        'start': start poition in R format (python format + 1),
        'end': end position,
        'seq': the porbable ORF itself,
        'probability': the probability of the highest possible candidate
        'max_length_exceeded': a boolean flag which indicates if at least one of the candidates was longer than MAX_LENGTH
    }
    (optionally only the one with the highest probability might be returned)

    To achieve this the function makes the following steps:
    1) Generate all possible candidates
    2) Keep only those that are shorter than MAX_LENGTH
    3) Calculate the porbability for each of them
    4) Return the one with the highest probability
    """
    sequence = sequence.lower()
    candidates = extract_candidates(seq=sequence, start_codons=START_CODONS, stop_codons=STOP_CODONS, extension=EXTENSION)

    if truncated:
        start_position = 3
        end_position = -3
        max_length_addition = 6
    else:
        start_position = 0
        end_position = None
        max_length_addition = 0
    candidates_filtered = [candidate for candidate in candidates if len(candidate['seq']) <= MAX_LENGTH + max_length_addition]

    if len(candidates_filtered) < len(candidates):
        max_length_exceeded = True
        candidates = candidates_filtered
    else:
        max_length_exceeded = False

    for candidate in candidates:
        candidate['probability'] = predict(text=[candidate['seq'][start_position:end_position]],
                                           model=model,
                                           data_loader=data_loader)[0]
        if not include_seq:
            candidate.pop('seq')
    if return_best:
        return max(candidates, key=lambda item: item['probability'], default={})
    elif return_top is not None:
        return sorted(candidates, key=lambda item: item['probability'], reverse=True)[:return_top]
    else:
        return candidates


@app.route('/orf_coordinates', methods=['POST'])
def process_request():
    data = request.get_json(force=True).items()
    data = {key: str(value) for key, value in data if len(str(value)) > 0}
    if len(data) > 0:
        respond = {}
        for key, value in data.items():
            respond[key] = find_orf(sequence=value, truncated=TRUNCATED)
        return jsonify(respond)
    else:
        return jsonify([])


@app.route('/test_connection', methods=['GET'])
def test():
    return 200, 'connection ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)