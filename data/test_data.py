from collections import Counter

def check_line_uniqueness(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    line_counts = Counter(line.strip() for line in lines)
    
    unique_lines = sum(1 for count in line_counts.values() if count == 1)
    non_unique_lines = len(lines) - unique_lines
    
    print(f"Total lines: {len(lines)}")
    print(f"Unique lines: {unique_lines}")
    print(f"Non-unique lines: {non_unique_lines}")


def check_line_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    strip_lines = [line.strip() for line in lines]    
    seq_count = 0
    total_len = 0
    more_than_256 = 0
    more_than_512 = 0
    more_than_512_average = 0

    for line in strip_lines:
        seq_count += 1
        total_len += len(line)
        if(len(line) > 256):
            more_than_256 += 1
        if(len(line) > 512):
            more_than_512 += 1
            more_than_512_average += len(line)


    print(f"Average line length: {total_len/seq_count:.2f}")
    print(f"More than 256: {more_than_256}")
    print(f"More than 512: {more_than_512}")

    #print(f"More than 512 average: {more_than_512_average/more_than_512:.2f}")
file_path = 'dnmt_unformatted.txt'
check_line_uniqueness(file_path)
check_line_lengths(file_path)


