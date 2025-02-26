from datasets import Dataset
from itertools import product

def num2chr(num: int) -> str:
    return str(num) if num < 10 else chr(ord('a') + num - 10)

def chr2num(char: str) -> int:
    return int(char) if char.isdigit() else ord(char) - ord('a') + 10

def generate_all_combinations(s_num_range: tuple[int, int], s_len_range: tuple[int, int], b_range: tuple[int, int]):
    assert (s_len_range[0] <= s_len_range[1]) and s_len_range[0] > 0, "Invalid string length range"
    assert (b_range[0] <= b_range[1]) and b_range[0] >= 2, "Invalid base range - must be at least 2"
    assert (s_num_range[0] <= s_num_range[1]) and s_num_range[0] >= 0

    valid_digits = range(s_num_range[0], s_num_range[1] + 1)
    all_examples = []
    for length in range(s_len_range[0], s_len_range[1] + 1):
        for combo in product(valid_digits, repeat=length):
            s = ''.join(num2chr(d) for d in combo).lstrip('0')
            if len(s) != length:
                continue

            max_digit = max(chr2num(d) for d in s)
            min_valid_base = max(b_range[0], max_digit + 1)
            for b in range(min_valid_base, b_range[1] + 1):
                try:
                    t = int(s, b)
                    all_examples.append((s, b, t))
                except ValueError:
                    continue

    return Dataset.from_dict({
        'S': [s for s, _, _ in all_examples],
        'B': [b for _, b, _ in all_examples],
        'T': [t for _, _, t in all_examples]
    })

# TEST

def count_possible_combinations(s_num_range, s_len_range, b_range):
    cnt = 0
    na, nb = s_num_range
    l_min, l_max = s_len_range
    b_min, b_max = b_range
    for b_i in range(b_min, b_max + 1):
        _nb = min(nb, b_i - 1)
        n = _nb - na + 1
        m = n - (1 if na == 0 else 0)
        if m < 0 or n < 0:
            continue

        if n == 1:
            s_cnt = m * (l_max - l_min + 1)
        else:
            first_term = m * (n ** (l_min - 1))
            num_terms = l_max - l_min + 1
            s_cnt = first_term * (n ** num_terms - 1) // (n - 1)
        cnt += s_cnt
    return cnt

def test_dataset(my_dataset, s_num_range, s_len_range, b_range):
    t_set = set()
    for d in my_dataset:
        assert int(d['S'], d['B']) == d['T']
        t_set.add((d['S'], d['B']))
    assert len(my_dataset) == len(t_set)
    assert len(my_dataset) == count_possible_combinations(s_num_range, s_len_range, b_range)


if __name__ == '__main__':
    s_num_range = (0, 1)
    for sl_max, br_max in product([4, 8, 16], [4, 8, 16]):
        s_len_range, b_range = (2, sl_max), (2, br_max)
        dataset_url = f"sdpkjc/NumBase-N{num2chr(s_num_range[0])}{num2chr(s_num_range[1])}-S{num2chr(s_len_range[0])}{num2chr(s_len_range[1])}-B{num2chr(b_range[0])}{num2chr(b_range[1])}"

        dataset = generate_all_combinations(s_num_range, s_len_range, b_range)
        dataset = dataset.shuffle()
        # print(dataset)
        # print(dataset[-10:-1])
        test_dataset(dataset, s_num_range, s_len_range, b_range)
        print(dataset_url, "\n")
        dataset.push_to_hub(dataset_url)
