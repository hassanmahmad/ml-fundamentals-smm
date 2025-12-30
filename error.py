def compute_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

def multiply_numbers(a, b, c):
    result = a * b * c
    return result

def find_largest(numbers):
    largest = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > largest:
            largest = numbers[i]
    return largest

def get_number_input():
    num = int(input("Enter a number: "))
    return num

# Main program
numbers = [3, 0, 6, 2]

print("The average is:", compute_average(numbers))

a = 5
b = "10"
c = 3
print("The product is:", multiply_numbers(a, b, c))

print("The largest number is:", find_largest(numbers))

user_number = get_number_input()
print("You entered:", user_number)