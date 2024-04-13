# This file's implementation is for 32-bit integers
# For 64-bit long integers, the same logic applies
# However, note that for long integers, you should use 1L instead of 1
# For example, num & (1L << 48) is correct for long integers

def print_binary(num):
    for i in range(31, -1, -1):
        print((num & (1 << i)) == 0 and "0" or "1", end="") 
        # 注意这里不要用 if num & (1 << i)) != 1:  因为如果i位上是1，但其它位置不为0 那么num & (1 >> i)的结果很可能不是1
    print("Done")

if __name__ == "__main__":
    # Non-negative number
    a = 78
    print(a)
    print_binary(a)
    print("===a===")

    # Negative number
    b = -6
    print(b)
    print_binary(b)
    print("===b===")

    # Define variable using binary literal
    c = 0b1001110
    print(c)
    print_binary(c)
    print("===c===")

    # Define variable using hexadecimal literal
    d = 0x4e
    print(d)
    print_binary(d)
    print("===d===")

    # Bitwise NOT (~)
    print(a)
    print_binary(a)
    print_binary(~a)
    e = ~a + 1
    print(e)
    print_binary(e)
    print("===e===")

    # Minimum int and long value
    f = -2**31
    print(f)
    print_binary(f)
    print(-f)
    print_binary(-f)
    print(~f + 1)
    print_binary(~f + 1)
    print("===f===")

    # Bitwise OR (|), AND (&), XOR (^)
    g = 0b0001010
    h = 0b0001100
    print(bin(g | h))
    print(bin(g & h))
    print(bin(g ^ h))
    print("===g, h===")

    # Bitwise shift operators
    i = 0b0011010
    print(bin(i))
    print(bin(i << 1))
    print(bin(i << 2))
    print(bin(i << 3))
    print("===i << ===")

    # Right shift (>>) and zero-fill right shift (>>>)
    # For positive numbers, both have the same effect
    # However, for negative numbers, the effect is different
    j = 0b11110000000000000000000000000000
    print(bin(j))
    print(bin(j >> 2))
    print("===j >> >>>===")

    # Multiply and divide by 2
    k = 10
    print(k)
    print(k << 1)
    print(k << 2)
    print(k << 3)
    print(k >> 1)
    print(k >> 2)
    print(k >> 3)
    print("===k===")