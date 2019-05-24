from predict import load_model, predict
from utils.seq_utils import extract_candidates
from flask import Flask, request, jsonify
from constants import MAX_LENGTH, START_CODONS, STOP_CODONS, BEST_MODEL_PATH

app = Flask(__name__)
model, data_loader = load_model(BEST_MODEL_PATH)


def prepare_report(sequence):
    """
    This function takes a single sequence and returns a dictionary in the following format:
    {
        'start': start poition in R format (python format + 1),
        'end': end position,
        'seq': the porbable ORF itself,
        'probability': the probability of the highest possible candidate
        'max_length_exceeded': a boolean flag which indicates if at least one of the candidates was longer than MAX_LENGTH
    }
    To achieve this the function makes the following steps:
    1) Generate all possible candidates
    2) Keep only those that are shorter than MAX_LENGTH
    3) Calculate the porbability for each of them
    4) Return the one with the highest probability
    """
    sequence = sequence.lower()
    candidates = extract_candidates(seq=sequence, start_codons=START_CODONS, stop_codons=STOP_CODONS)
    candidates_filtered = [candidate for candidate in candidates if len(candidate['seq']) <= MAX_LENGTH]
    if len(candidates_filtered) < len(candidates):
        max_length_exceeded = True
        candidates = candidates_filtered
    else:
        max_length_exceeded = False

    for candidate in candidates:
        candidate['probability'] = predict(text=[candidate['seq']], model=model, data_loader=data_loader)
    # TODO: Finish the algortihm: return the candidate with the highst probability


@app.route('/orf_coordinates', methods=['POST'])
def process_request():
    data = request.get_json(force=True).items()
    data = {key: str(value) for key, value in data if len(str(value)) > 0}
    if len(data) > 0:
        keys = list(data.keys())
        values = list(data.values())
        predictions = predict(values, model, data_loader)
        respond = {key: prediction for key, prediction in zip(keys, predictions)}
        return jsonify(respond)
    else:
        return jsonify([])


@app.route('/test_connection', methods=['GET'])
def test():
    return 'connection ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)