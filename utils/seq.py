import re


def extract_candidates(seq, start_codons, stop_codons, min_length=9, extension=100):
    seq = seq.lower()
    start_codons = [codon.lower() for codon in start_codons]
    stop_codons = [codon.lower() for codon in stop_codons]
    
    start_codons_coordinates = [match.start() for codon in start_codons
                                                    for match in re.finditer(codon, seq) if match.start() >= extension + 1]
    stop_codons_coordinates = [match.end() for codon in stop_codons
                                                   for match in re.finditer(codon, seq) if len(seq) - match.end() >= extension]
    candidates_coordinates = [(start, end) for start in start_codons_coordinates 
                                                  for end in stop_codons_coordinates
                                                          if ((end - start) >= min_length and (end - start) % 3 == 0)]
    # the start and end positions are returned in python style!
    candidates = [{'seq': seq[start - extension: end + extension], 
                   'start': start, 
                   'end': end}
                    for start, end in candidates_coordinates]
    return candidates