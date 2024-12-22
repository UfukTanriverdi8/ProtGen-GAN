from Bio.Align import PairwiseAligner

def calculate_sequence_identity(seq1, seq2):
    """
    Calculate the percentage identity between two amino acid sequences using PairwiseAligner.
    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
    Returns:
        float: Percentage identity between the two sequences.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"  # Perform global alignment
    alignment = aligner.align(seq1, seq2)[0]  # Get the best alignment

    # Extract aligned sequences
    aligned_seq1 = alignment.target
    aligned_seq2 = alignment.query

    # Count matches
    matches = sum(c1 == c2 for c1, c2 in zip(aligned_seq1, aligned_seq2))
    
    # Calculate sequence identity
    percent_identity = matches / max(len(seq1), len(seq2)) * 100
    return percent_identity
